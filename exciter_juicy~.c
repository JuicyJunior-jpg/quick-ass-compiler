// exciter_juicy~.c
// A physical-modelling exciter for PlugData/Pure Data
// Juicy spec: ADSR (noise only), click type (impulse/DC), click volume,
// noise gain, filter (lp/hp/bp, 0-1 mapped 1Hz..~Nyquist), Hardness (anti-sparse blend).
// DC is gated by note on/off. ADSR is velocity-sensitive and time-accurate (ms).
// Smooths parameter changes to avoid zipper noise.
//
// Build: clang -O3 -DPD -std=c99 -fPIC -shared -o exciter_juicy~.pd_darwin exciter_juicy~.c
// Linux: clang -O3 -DPD -std=c99 -fPIC -shared -o exciter_juicy~.pd_linux exciter_juicy~.c -lm
// Windows (MinGW): clang -O3 -DPD -std=c99 -shared -o exciter_juicy~.dll exciter_juicy~.c -lm
//
// Object name in Pd: [exciter_juicy~]
//
// Inlets (floats; smoothed internally):
//   1: Attack_ms
//   2: Decay_ms
//   3: Sustain (0..1)
//   4: Release_ms
//   5: ClickVolume (0..1)         // if IMPULSE: scales 1-sample impulse; if DC: sets DC level while note held
//   6: NoiseGain (0..1)           // amplitude of noise path (after ADSR*velocity)
//   7: FilterNorm (0..1)          // mapped 1 Hz .. ~0.45*sr (log scale)
//   8: Hardness (0..1)            // blends anti-sparse + color layers into base noise
//
// Messages:
//   note <pitch> <velocity>       // velocity 0..127; >0 = note on, 0 = note off
//   gate <0/1>                    // alternative to note; 1 = on, 0 = off (velocity remains last set or 1)
//   vel <0..1>                    // set velocity scaler directly (used by ADSR on noise path)
//   impulse                       // click mode = IMPULSE (1-sample impulse on note on)
//   dc                            // click mode = DC (steady DC while note held; smoothed on/off)
//   filter lp|hp|bp               // select filter mode
//
// Outlet:
//   1: signal (exciter output)
//
// Notes:
// - ADSR time accuracy: attack/decay/release durations computed in samples from ms values.
// - 0 ms attack/decay/release = instantaneous transition to target.
// - Parameter smoothing for continuous knobs uses a short time constant to avoid zipper noise
//   without affecting ADSR timing.
//
// Copyright (c) 2025
//

#include "m_pd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------- RNG (fast, per-instance xorshift32) ----------------
typedef struct {
    unsigned int state;
} rng32_t;

static inline void rng32_seed(rng32_t* r, unsigned int s) {
    if (s == 0) s = 0xA3C59AC3u;
    r->state = s;
}
static inline float rng32_nextf(rng32_t* r) {
    // xorshift32, returns float in (-1,1)
    unsigned int x = r->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r->state = x;
    // scale to (0,1), then to (-1,1)
    float u = (x / 4294967296.0f);
    return 2.0f * u - 1.0f;
}

// ---------------- State Variable Filter (simple Chamberlin) -----------
// Mode: 0=LP, 1=HP, 2=BP
typedef struct {
    float lp, bp; // states
    float f;      // normalized cutoff coefficient (0..1)
    float q;      // damping (typ ~0.5..1.5); lower = more resonance
    int mode;
} svf_t;

static inline void svf_init(svf_t* s) {
    s->lp = 0.0f;
    s->bp = 0.0f;
    s->f  = 0.0f;
    s->q  = 0.707f; // ~Butterworth
    s->mode = 0;
}
static inline void svf_set_mode(svf_t* s, int mode) { s->mode = mode; }
static inline void svf_set_cut(svf_t* s, float hz, float sr) {
    // Chamberlin uses f = 2*sin(pi*fc/sr); clamp for stability
    float x = (float)M_PI * (hz / sr);
    if (x > (float)M_PI * 0.49f) x = (float)M_PI * 0.49f;
    float f = 2.0f * sinf(x);
    if (f > 1.999f) f = 1.999f;
    if (f < 0.0f)   f = 0.0f;
    s->f = f;
}
static inline float svf_process(svf_t* s, float in) {
    // Chamberlin integrator form
    float hp = in - s->lp - s->q * s->bp;
    s->bp += s->f * hp;
    s->lp += s->f * s->bp;
    switch (s->mode) {
        case 0:  return s->lp; // low-pass
        case 1:  return hp;    // high-pass
        default: return s->bp; // band-pass
    }
}

// ---------------- Parameter smoother (1-pole) -------------------------
// y += a*(x - y);  with a = 1 - exp(-1/(tau*sr))
typedef struct {
    float y;
    float a; // coefficient
} smooth_t;

static inline void smooth_set_tau(smooth_t* s, float tau_ms, float sr) {
    float tau = (tau_ms <= 0.f ? 0.000001f : tau_ms * 0.001f);
    float a = 1.0f - expf(-1.0f / (tau * sr));
    if (a < 0.000001f) a = 0.000001f;
    if (a > 1.0f) a = 1.0f;
    s->a = a;
}
static inline void smooth_init(smooth_t* s, float init) {
    s->y = init;
    s->a = 0.001f;
}
static inline float smooth_tick(smooth_t* s, float x) {
    s->y += s->a * (x - s->y);
    return s->y;
}

// ---------------- ADSR (time-accurate in ms) --------------------------
typedef enum { ENV_IDLE=0, ENV_ATTACK, ENV_DECAY, ENV_SUSTAIN, ENV_RELEASE } env_stage_t;

typedef struct {
    env_stage_t stage;
    float sr;
    // times (ms)
    float attack_ms, decay_ms, release_ms;
    float sustain;     // 0..1
    // internals
    float value;       // current env value
    float vel_scale;   // 0..1, set on note-on
    // per-stage increments (recomputed on enter)
    float inc;         // linear increment per sample
    float target;      // target value for current stage
} adsr_t;

static inline void adsr_init(adsr_t* e, float sr) {
    e->stage = ENV_IDLE;
    e->sr = sr;
    e->attack_ms = 0.f;
    e->decay_ms = 0.f;
    e->release_ms = 0.f;
    e->sustain = 0.f;
    e->value = 0.f;
    e->vel_scale = 1.f;
    e->inc = 0.f;
    e->target = 0.f;
}

static inline void adsr_set_times(adsr_t* e, float att_ms, float dec_ms, float sus, float rel_ms) {
    e->attack_ms = (att_ms < 0.f ? 0.f : att_ms);
    e->decay_ms  = (dec_ms  < 0.f ? 0.f : dec_ms);
    e->sustain   = (sus < 0.f ? 0.f : (sus > 1.f ? 1.f : sus));
    e->release_ms= (rel_ms < 0.f ? 0.f : rel_ms);
}

// compute inc to reach target in exactly duration_ms (linear ramp)
static inline float _adsr_calc_inc(float current, float target, float duration_ms, float sr) {
    if (duration_ms <= 0.f) return (target - current); // jump immediately in one sample
    float n = (duration_ms * 0.001f) * sr;
    if (n < 1.f) n = 1.f;
    return (target - current) / n;
}

static inline void adsr_note_on(adsr_t* e, float vel01) {
    e->vel_scale = (vel01 < 0.f ? 0.f : (vel01 > 1.f ? 1.f : vel01));
    // start attack to 1.0
    e->stage = ENV_ATTACK;
    e->target = 1.0f;
    e->inc = _adsr_calc_inc(e->value, e->target, e->attack_ms, e->sr);
}

static inline void adsr_note_off(adsr_t* e) {
    e->stage = ENV_RELEASE;
    e->target = 0.0f;
    e->inc = _adsr_calc_inc(e->value, e->target, e->release_ms, e->sr);
}

static inline float adsr_tick(adsr_t* e) {
    switch (e->stage) {
        case ENV_ATTACK:
            e->value += e->inc;
            if ((e->inc >= 0.f && e->value >= e->target) || (e->inc < 0.f && e->value <= e->target)) {
                e->value = e->target;
                // move to decay
                e->stage = ENV_DECAY;
                e->target = e->sustain;
                e->inc = _adsr_calc_inc(e->value, e->target, e->decay_ms, e->sr);
            }
            break;
        case ENV_DECAY:
            e->value += e->inc;
            if ((e->inc >= 0.f && e->value <= e->target) || (e->inc < 0.f && e->value >= e->target)) {
                e->value = e->target;
                e->stage = ENV_SUSTAIN;
                e->inc = 0.f;
            }
            break;
        case ENV_SUSTAIN:
            // hold
            break;
        case ENV_RELEASE:
            e->value += e->inc;
            if ((e->inc <= 0.f && e->value <= e->target) || (e->inc > 0.f && e->value >= e->target)) {
                e->value = e->target;
                e->stage = ENV_IDLE;
                e->inc = 0.f;
            }
            break;
        default:
            // idle
            break;
    }
    return e->value * e->vel_scale;
}

// ---------------- Main object ----------------------------------------

typedef struct _exciter_juicy_tilde {
    t_object x_obj;
    t_outlet* x_out;

    // sample rate
    float sr;

    // 8 knob targets (set via float inlets), and smoothed/current values
    float atk_ms_t, dec_ms_t, sus_t, rel_ms_t;
    float click_vol_t, noise_gain_t, filter_norm_t, hardness_t;

    // smoothed params (for continuous knobs)
    smooth_t click_vol_s, noise_gain_s, filter_norm_s, hardness_s;

    // ADSR for noise
    adsr_t env;

    // RNG
    rng32_t rng;

    // filter
    svf_t svf;
    int filter_mode; // 0 LP,1 HP,2 BP

    // DC / Click mode
    int click_is_dc;  // 0=impulse, 1=dc
    int gate;         // current gate state (note held)
    float velocity01; // 0..1 normalized velocity

    // impulse one-shot
    float pending_impulse; // amplitude to add for exactly one sample

    // DC smoothing
    smooth_t dc_s;    // smoothed DC value (target set by gate and click_vol)
    float dc_target;  // target DC (click_vol when gate on and dc mode, else 0)

} t_exciter_juicy_tilde;

// Forward declarations
static t_int* exciter_juicy_tilde_perform(t_int* w);
static void exciter_juicy_tilde_dsp(t_exciter_juicy_tilde* x, t_signal** sp);
static void* exciter_juicy_tilde_new(void);
static void exciter_juicy_tilde_free(t_exciter_juicy_tilde* x);

// Methods
static void exciter_juicy_tilde_note(t_exciter_juicy_tilde* x, t_symbol* s, int argc, t_atom* argv);
static void exciter_juicy_tilde_gate(t_exciter_juicy_tilde* x, t_floatarg f);
static void exciter_juicy_tilde_vel(t_exciter_juicy_tilde* x, t_floatarg f);
static void exciter_juicy_tilde_impulse(t_exciter_juicy_tilde* x);
static void exciter_juicy_tilde_dc(t_exciter_juicy_tilde* x);
static void exciter_juicy_tilde_filtermode(t_exciter_juicy_tilde* x, t_symbol* s, int argc, t_atom* argv);

// Helpers
static inline float clamp01(float v) { return (v < 0.f ? 0.f : (v > 1.f ? 1.f : v)); }
static inline float mixf(float a, float b, float t) { return a + t * (b - a); }

// Perform
static t_int* exciter_juicy_tilde_perform(t_int* w) {
    t_exciter_juicy_tilde* x = (t_exciter_juicy_tilde*)(w[1]);
    int n = (int)(w[2]);
    t_sample* out = (t_sample*)(w[3]);

    float sr = x->sr;

    // update filter cutoff smoothing coeff once per block
    smooth_set_tau(&x->filter_norm_s, 10.0f, sr); // ~10ms smoothing for knobs
    smooth_set_tau(&x->click_vol_s,   5.0f,  sr);
    smooth_set_tau(&x->noise_gain_s,  5.0f,  sr);
    smooth_set_tau(&x->hardness_s,    10.0f, sr);
    smooth_set_tau(&x->dc_s,          5.0f,  sr);

    for (int i = 0; i < n; ++i) {
        // Smooth knob params
        float click_vol  = clamp01(smooth_tick(&x->click_vol_s,  x->click_vol_t));
        float noise_gain = clamp01(smooth_tick(&x->noise_gain_s, x->noise_gain_t));
        float hardness   = clamp01(smooth_tick(&x->hardness_s,   x->hardness_t));
        float fnorm      = clamp01(smooth_tick(&x->filter_norm_s,x->filter_norm_t));

        // Map filter norm (0..1) to Hz (1 Hz .. 0.45*sr) on log scale
        float fmin = 1.0f;
        float fmax = 0.45f * sr;
        // Avoid log(0)
        float logmin = logf(fmin);
        float logmax = logf(fmax);
        float cutoff = expf(logmin + (logmax - logmin) * fnorm);
        if (cutoff < 1.0f) cutoff = 1.0f;
        if (cutoff > fmax) cutoff = fmax;
        svf_set_cut(&x->svf, cutoff, sr);
        x->svf.q = 0.707f; // fixed damping; could be exposed later

        // Envelope tick (noise only)
        // ADSR times come directly from targets (time-accuracy). If user moved knobs mid-phase,
        // the next stage transition recalculates increments with the new ms values.
        // We don't smooth times to keep exact durations by ms.
        x->env.sustain = clamp01(x->sus_t);
        x->env.attack_ms  = (x->atk_ms_t < 0.f ? 0.f : x->atk_ms_t);
        x->env.decay_ms   = (x->dec_ms_t < 0.f ? 0.f : x->dec_ms_t);
        x->env.release_ms = (x->rel_ms_t < 0.f ? 0.f : x->rel_ms_t);
        float env = adsr_tick(&x->env); // already includes velocity scale

        // Base white noise
        float n0 = rng32_nextf(&x->rng); // -1..1

        // Anti-sparse layer: fold & square, then rescale to [-1,1]
        float a1 = fabsf(n0);           // 0..1
        float folded = 2.0f * a1 - 1.0f; // -1..1, denser distribution
        float squared = (n0 >= 0.f ? n0*n0 : -(n0*n0)); // keep sign for texture
        float anti = 0.5f * folded + 0.5f * squared;   // blend two shapes

        float noise_mix = mixf(n0, anti, hardness);
        float noise_out = noise_gain * env * noise_mix;

        // Click/DC path
        float click_dc = 0.0f;
        if (x->click_is_dc) {
            // DC active only when gate is on
            x->dc_target = (x->gate ? click_vol : 0.0f);
            click_dc = smooth_tick(&x->dc_s, x->dc_target);
        } else {
            // impulse: one-sample transient on note-on
            click_dc = x->pending_impulse;
            x->pending_impulse = 0.0f; // consume
        }

        // Sum paths
        float sum = noise_out + click_dc;

        // Filter
        x->svf.mode = x->filter_mode;
        float y = svf_process(&x->svf, sum);

        out[i] = y;
    }

    return (w + 4);
}

// DSP add
static void exciter_juicy_tilde_dsp(t_exciter_juicy_tilde* x, t_signal** sp) {
    x->sr = sp[0]->s_sr > 0 ? sp[0]->s_sr : 48000.0f;
    adsr_init(&x->env, x->sr);
    // keep current velocity
    x->env.vel_scale = x->velocity01;
    // (Re)initialize filter states to avoid pops
    svf_init(&x->svf);
    // set smoother initial coeffs
    smooth_set_tau(&x->click_vol_s,   5.0f,  x->sr);
    smooth_set_tau(&x->noise_gain_s,  5.0f,  x->sr);
    smooth_set_tau(&x->filter_norm_s, 10.0f, x->sr);
    smooth_set_tau(&x->hardness_s,    10.0f, x->sr);
    smooth_set_tau(&x->dc_s,          5.0f,  x->sr);
    dsp_add(exciter_juicy_tilde_perform, 3, x, sp[0]->s_n, sp[0]->s_vec);
}

// note <pitch> <velocity>
static void exciter_juicy_tilde_note(t_exciter_juicy_tilde* x, t_symbol* s, int argc, t_atom* argv) {
    (void)s;
    if (argc < 2) return;
    float pitch = atom_getfloatarg(0, argc, argv); (void)pitch;
    float vel   = atom_getfloatarg(1, argc, argv);
    int ivel = (int)vel;
    if (ivel > 0) {
        x->velocity01 = (vel <= 1.f) ? vel : (vel / 127.f);
        adsr_note_on(&x->env, x->velocity01);
        x->gate = 1;
        // prepare impulse if mode is impulse
        if (!x->click_is_dc) {
            float amp = clamp01(x->click_vol_t);
            x->pending_impulse = amp;
        } else {
            // dc target will be updated in perform
            x->dc_target = clamp01(x->click_vol_t);
        }
    } else {
        adsr_note_off(&x->env);
        x->gate = 0;
        // dc will smooth to 0 if in dc mode
        x->dc_target = 0.0f;
    }
}

// gate <0/1>
static void exciter_juicy_tilde_gate(t_exciter_juicy_tilde* x, t_floatarg f) {
    if (f != 0) {
        adsr_note_on(&x->env, x->velocity01);
        x->gate = 1;
        if (!x->click_is_dc) {
            x->pending_impulse = clamp01(x->click_vol_t);
        } else {
            x->dc_target = clamp01(x->click_vol_t);
        }
    } else {
        adsr_note_off(&x->env);
        x->gate = 0;
        x->dc_target = 0.0f;
    }
}

// vel <0..1>
static void exciter_juicy_tilde_vel(t_exciter_juicy_tilde* x, t_floatarg f) {
    float v = clamp01(f);
    x->velocity01 = v;
    x->env.vel_scale = v;
}

// impulse
static void exciter_juicy_tilde_impulse(t_exciter_juicy_tilde* x) {
    x->click_is_dc = 0;
}
// dc
static void exciter_juicy_tilde_dc(t_exciter_juicy_tilde* x) {
    x->click_is_dc = 1;
}

// filter lp|hp|bp
static void exciter_juicy_tilde_filtermode(t_exciter_juicy_tilde* x, t_symbol* s, int argc, t_atom* argv) {
    (void)s;
    if (argc < 1) return;
    t_symbol* mode = atom_getsymbolarg(0, argc, argv);
    if (mode == gensym("lp") || mode == gensym("lowpass")) {
        x->filter_mode = 0;
    } else if (mode == gensym("hp") || mode == gensym("highpass")) {
        x->filter_mode = 1;
    } else if (mode == gensym("bp") || mode == gensym("bandpass")) {
        x->filter_mode = 2;
    }
}

// New / Free
static void* exciter_juicy_tilde_new(void) {
    t_exciter_juicy_tilde* x = (t_exciter_juicy_tilde*)pd_new(exciter_juicy_tilde_class);
    if (!x) return NULL;

    x->x_out = outlet_new(&x->x_obj, &s_signal);
    x->sr = sys_getsr();
    if (x->sr <= 0) x->sr = 48000.0f;

    // Defaults
    x->atk_ms_t = 0.f;
    x->dec_ms_t = 0.f;
    x->sus_t    = 0.7f;
    x->rel_ms_t = 50.f;
    x->click_vol_t   = 0.5f;
    x->noise_gain_t  = 0.8f;
    x->filter_norm_t = 0.5f;
    x->hardness_t    = 0.0f;

    smooth_init(&x->click_vol_s,   x->click_vol_t);
    smooth_init(&x->noise_gain_s,  x->noise_gain_t);
    smooth_init(&x->filter_norm_s, x->filter_norm_t);
    smooth_init(&x->hardness_s,    x->hardness_t);
    smooth_init(&x->dc_s,          0.0f);

    rng32_seed(&x->rng, 2222u);
    svf_init(&x->svf);
    adsr_init(&x->env, x->sr);

    x->filter_mode = 0; // LP
    x->click_is_dc = 0; // impulse by default
    x->gate = 0;
    x->velocity01 = 1.0f;
    x->pending_impulse = 0.0f;
    x->dc_target = 0.0f;

    // Create 8 float inlets (knobs)
    floatinlet_new(&x->x_obj, &x->atk_ms_t);
    floatinlet_new(&x->x_obj, &x->dec_ms_t);
    floatinlet_new(&x->x_obj, &x->sus_t);
    floatinlet_new(&x->x_obj, &x->rel_ms_t);
    floatinlet_new(&x->x_obj, &x->click_vol_t);
    floatinlet_new(&x->x_obj, &x->noise_gain_t);
    floatinlet_new(&x->x_obj, &x->filter_norm_t);
    floatinlet_new(&x->x_obj, &x->hardness_t);

    return (void*)x;
}

static void exciter_juicy_tilde_free(t_exciter_juicy_tilde* x) {
    (void)x;
}

// Setup
static t_class* exciter_juicy_tilde_class;

void exciter_juicy_tilde_setup(void) {
    t_class* c = class_new(gensym("exciter_juicy~"),
                           (t_newmethod)exciter_juicy_tilde_new,
                           (t_method)exciter_juicy_tilde_free,
                           sizeof(t_exciter_juicy_tilde),
                           CLASS_DEFAULT,
                           0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_dsp, gensym("dsp"), A_CANT, 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_note, gensym("note"), A_GIMME, 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_gate, gensym("gate"), A_FLOAT, 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_vel,  gensym("vel"),  A_FLOAT, 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_impulse, gensym("impulse"), 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_dc,      gensym("dc"), 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_filtermode, gensym("filter"), A_GIMME, 0);

    CLASS_MAINSIGNALIN(c, t_exciter_juicy_tilde, sr); // dummy to allow signal inlet if needed
    class_sethelpsymbol(c, gensym("exciter_juicy~-help"));

    exciter_juicy_tilde_class = c;
}
