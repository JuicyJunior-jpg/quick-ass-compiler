// exciter_juicy~.c (STEREO + Hardness upgrade, stereo dsp fix)
// - Stereo output (independent noise per ear for width).
// - Hardness adds extra noise layers + nonlinearity with gain compensation.
// - ADSR (noise-only, ms, velocity-aware).
// - Click = 1-sample impulse OR DC (gated on note on/off, smoothed).
// - Filter: LP/HP/BP per channel, 0–1 → 1Hz..~0.45*sr (log).

#include "m_pd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static t_class* exciter_juicy_tilde_class;

// ---------------- RNG (xorshift32) ----------------
typedef struct { unsigned int state; } rng32_t;
static inline void rng32_seed(rng32_t* r, unsigned int s){ if(!s) s=0xA3C59AC3u; r->state=s; }
static inline float rng32_nextf(rng32_t* r){
    unsigned int x=r->state; x^=x<<13; x^=x>>17; x^=x<<5; r->state=x;
    float u = (x / 4294967296.0f); return 2.f*u - 1.f;
}

// ---------------- State Variable Filter (Chamberlin) -----------
typedef struct { float lp, bp, f, q; int mode; } svf_t; // 0 LP, 1 HP, 2 BP
static inline void svf_init(svf_t* s){ s->lp=s->bp=0.f; s->f=0.f; s->q=0.707f; s->mode=0; }
static inline void svf_set_mode(svf_t* s, int mode){ s->mode=mode; }
static inline void svf_set_cut(svf_t* s, float hz, float sr){
    float x = (float)M_PI * (hz/sr); if(x>(float)M_PI*0.49f) x = (float)M_PI*0.49f;
    float f = 2.f * sinf(x); if(f>1.999f) f=1.999f; if(f<0.f) f=0.f; s->f=f;
}
static inline float svf_process(svf_t* s, float in){
    float hp = in - s->lp - s->q*s->bp;
    s->bp += s->f*hp;
    s->lp += s->f*s->bp;
    if(s->mode==0) return s->lp;
    else if(s->mode==1) return hp;
    else return s->bp;
}

// ---------------- 1-pole smoother -----------------
typedef struct { float y,a; } smooth_t;
static inline void smooth_set_tau(smooth_t* s, float tau_ms, float sr){
    float tau = (tau_ms<=0.f? 0.000001f : tau_ms*0.001f);
    float a = 1.f - expf(-1.f/(tau*sr));
    if(a<0.000001f) a=0.000001f; if(a>1.f) a=1.f; s->a=a;
}
static inline void smooth_init(smooth_t* s, float init){ s->y=init; s->a=0.001f; }
static inline float smooth_tick(smooth_t* s, float x){ s->y += s->a*(x - s->y); return s->y; }

// ---------------- ADSR (time-accurate ms) ----------------------
typedef enum { ENV_IDLE=0, ENV_ATTACK, ENV_DECAY, ENV_SUSTAIN, ENV_RELEASE } env_stage_t;
typedef struct {
    env_stage_t stage; float sr;
    float attack_ms, decay_ms, release_ms, sustain;
    float value, vel_scale, inc, target;
} adsr_t;
static inline void adsr_init(adsr_t* e, float sr){
    e->stage=ENV_IDLE; e->sr=sr; e->attack_ms=e->decay_ms=e->release_ms=0.f; e->sustain=0.f;
    e->value=0.f; e->vel_scale=1.f; e->inc=0.f; e->target=0.f;
}
static inline float _adsr_calc_inc(float cur,float tgt,float dur_ms,float sr){
    if(dur_ms<=0.f) return (tgt-cur);
    float n = (dur_ms*0.001f)*sr; if(n<1.f) n=1.f; return (tgt-cur)/n;
}
static inline void adsr_note_on(adsr_t* e, float vel01){
    e->vel_scale = (vel01<0?0:(vel01>1?1:vel01));
    e->stage=ENV_ATTACK; e->target=1.f; e->inc=_adsr_calc_inc(e->value,e->target,e->attack_ms,e->sr);
}
static inline void adsr_note_off(adsr_t* e){
    e->stage=ENV_RELEASE; e->target=0.f; e->inc=_adsr_calc_inc(e->value,e->target,e->release_ms,e->sr);
}
static inline float adsr_tick(adsr_t* e){
    switch(e->stage){
        case ENV_ATTACK:
            e->value += e->inc;
            if( (e->inc>=0.f && e->value>=e->target) || (e->inc<0.f && e->value<=e->target) ){
                e->value=e->target; e->stage=ENV_DECAY;
                e->target=e->sustain; e->inc=_adsr_calc_inc(e->value,e->target,e->decay_ms,e->sr);
            } break;
        case ENV_DECAY:
            e->value += e->inc;
            if( (e->inc>=0.f && e->value>=e->target) || (e->inc<0.f && e->value<=e->target) ){
                e->value=e->target; e->stage=ENV_SUSTAIN; e->inc=0.f;
            } break;
        case ENV_SUSTAIN: break;
        case ENV_RELEASE:
            e->value += e->inc;
            if( (e->inc<=0.f && e->value<=e->target) || (e->inc>0.f && e->value>=e->target) ){
                e->value=e->target; e->stage=ENV_IDLE; e->inc=0.f;
            } break;
        default: break;
    }
    return e->value * e->vel_scale;
}

// ---------------- Main object (stereo) --------------------
typedef struct _exciter_juicy_tilde {
    t_object x_obj; t_outlet* x_outL; t_outlet* x_outR; float sr;

    // targets
    float atk_ms_t, dec_ms_t, sus_t, rel_ms_t;
    float click_vol_t, noise_gain_t, filter_norm_t, hardness_t;
    // smoothed
    smooth_t click_vol_s, noise_gain_s, filter_norm_s, hardness_s;
    // ADSR
    adsr_t env;
    // RNG (independent per ear)
    rng32_t rngL, rngR;
    // filter per ear
    svf_t svfL, svfR; int filter_mode;

    // pre-emphasis LP state for high-detail layer
    float n_lpL, n_lpR; float n_lp_a;

    // click/DC
    int click_is_dc, gate; float velocity01;
    float pending_impulse;
    smooth_t dc_s; float dc_target;
} t_exciter_juicy_tilde;

// helpers
static inline float clamp01(float v){ return v<0.f?0.f:(v>1.f?1.f:v); }
static inline float mixf(float a,float b,float t){ return a + t*(b-a); }
static inline float tanh_norm(float x, float drive){
    float d = (drive<1.f?1.f:drive);
    float y = tanhf(d*x);
    float norm = tanhf(d);
    return (norm>0.f? y/norm : y);
}

static t_int* exciter_juicy_tilde_perform(t_int* w){
    t_exciter_juicy_tilde* x=(t_exciter_juicy_tilde*)(w[1]);
    int n=(int)(w[2]);
    t_sample* outL=(t_sample*)(w[3]);
    t_sample* outR=(t_sample*)(w[4]);
    float sr=x->sr;

    smooth_set_tau(&x->filter_norm_s,10.f,sr);
    smooth_set_tau(&x->click_vol_s,  5.f,sr);
    smooth_set_tau(&x->noise_gain_s, 5.f,sr);
    smooth_set_tau(&x->hardness_s,  10.f,sr);
    smooth_set_tau(&x->dc_s,         5.f,sr);

    for(int i=0;i<n;++i){
        float click_vol  = clamp01(smooth_tick(&x->click_vol_s,  x->click_vol_t));
        float noise_gain = clamp01(smooth_tick(&x->noise_gain_s, x->noise_gain_t));
        float hardness   = clamp01(smooth_tick(&x->hardness_s,   x->hardness_t));
        float fnorm      = clamp01(smooth_tick(&x->filter_norm_s,x->filter_norm_t));

        float fmin=1.f, fmax=0.45f*sr;
        float cutoff = expf(logf(fmin) + (logf(fmax)-logf(fmin))*fnorm);
        if(cutoff<1.f) cutoff=1.f; if(cutoff>fmax) cutoff=fmax;
        svf_set_cut(&x->svfL, cutoff, sr);
        svf_set_cut(&x->svfR, cutoff, sr);
        x->svfL.q = x->svfR.q = 0.707f;
        x->svfL.mode = x->svfR.mode = x->filter_mode;

        x->env.sustain   = clamp01(x->sus_t);
        x->env.attack_ms = (x->atk_ms_t<0?0:x->atk_ms_t);
        x->env.decay_ms  = (x->dec_ms_t<0?0:x->dec_ms_t);
        x->env.release_ms= (x->rel_ms_t<0?0:x->rel_ms_t);
        float env = adsr_tick(&x->env);

        float nL = rng32_nextf(&x->rngL);
        float nR = rng32_nextf(&x->rngR);

        float foldedL = 2.f*fabsf(nL)-1.f;
        float foldedR = 2.f*fabsf(nR)-1.f;
        float sqL = (nL>=0.f? nL*nL : -(nL*nL));
        float sqR = (nR>=0.f? nR*nR : -(nR*nR));
        float antiL = 0.5f*foldedL + 0.5f*sqL;
        float antiR = 0.5f*foldedR + 0.5f*sqR;

        float a = x->n_lp_a;
        x->n_lpL += a * (nL - x->n_lpL);
        x->n_lpR += a * (nR - x->n_lpR);
        float hiL = nL - x->n_lpL;
        float hiR = nR - x->n_lpR;

        float denseL = nL + hardness*(0.6f*antiL + 0.4f*hiL);
        float denseR = nR + hardness*(0.6f*antiR + 0.4f*hiR);

        float drive = 1.f + 4.f*hardness; // 1..5
        float shapedL = tanh_norm(denseL, drive);
        float shapedR = tanh_norm(denseR, drive);

        float noiseL = noise_gain * env * shapedL;
        float noiseR = noise_gain * env * shapedR;

        float click_dc = 0.f;
        if(x->click_is_dc){
            x->dc_target = (x->gate? click_vol : 0.f);
            click_dc = smooth_tick(&x->dc_s, x->dc_target);
        } else {
            click_dc = x->pending_impulse; x->pending_impulse=0.f;
        }

        float yL = svf_process(&x->svfL, noiseL + click_dc);
        float yR = svf_process(&x->svfR, noiseR + click_dc);

        outL[i]=yL; outR[i]=yR;
    }
    return (w+5);
}

static void exciter_juicy_tilde_dsp(t_exciter_juicy_tilde* x, t_signal** sp){
    x->sr = sp[0]->s_sr>0? sp[0]->s_sr : 48000.f;
    adsr_init(&x->env, x->sr); x->env.vel_scale = x->velocity01;
    svf_init(&x->svfL); svf_init(&x->svfR);
    float fc = 1000.f;
    x->n_lp_a = 1.f - expf(-2.f*(float)M_PI*fc / x->sr);
    x->n_lpL = x->n_lpR = 0.f;

    smooth_set_tau(&x->click_vol_s, 5.f, x->sr);
    smooth_set_tau(&x->noise_gain_s,5.f, x->sr);
    smooth_set_tau(&x->filter_norm_s,10.f,x->sr);
    smooth_set_tau(&x->hardness_s,10.f,x->sr);
    smooth_set_tau(&x->dc_s,5.f,x->sr);

    // IMPORTANT: because we used CLASS_MAINSIGNALIN, Pd gives us one signal inlet.
    // So the signal vector order is: [in0], [out0], [out1].
    // Wire our outs to sp[1] (L) and sp[2] (R).
    dsp_add(exciter_juicy_tilde_perform,4,x,sp[0]->s_n,sp[1]->s_vec,sp[2]->s_vec);
}

// messages
static void exciter_juicy_tilde_note(t_exciter_juicy_tilde* x, t_symbol* s, int argc, t_atom* argv){
    (void)s; if(argc<2) return;
    float pitch = atom_getfloatarg(0,argc,argv); (void)pitch;
    float velf  = atom_getfloatarg(1,argc,argv);
    int ivel = (int)velf;
    if(ivel>0){
        x->velocity01 = (velf<=1.f? velf : (velf/127.f));
        adsr_note_on(&x->env, x->velocity01);
        x->gate=1;
        if(!x->click_is_dc){
            x->pending_impulse = clamp01(x->click_vol_t);
        } else {
            x->dc_target = clamp01(x->click_vol_t);
        }
    } else {
        adsr_note_off(&x->env);
        x->gate=0; x->dc_target=0.f;
    }
}
static void exciter_juicy_tilde_gate(t_exciter_juicy_tilde* x, t_floatarg f){
    if(f!=0){
        adsr_note_on(&x->env, x->velocity01); x->gate=1;
        if(!x->click_is_dc) x->pending_impulse=clamp01(x->click_vol_t);
        else x->dc_target=clamp01(x->click_vol_t);
    } else {
        adsr_note_off(&x->env); x->gate=0; x->dc_target=0.f;
    }
}
static void exciter_juicy_tilde_vel(t_exciter_juicy_tilde* x, t_floatarg f){
    float v=clamp01(f); x->velocity01=v; x->env.vel_scale=v;
}
static void exciter_juicy_tilde_impulse(t_exciter_juicy_tilde* x){ x->click_is_dc=0; }
static void exciter_juicy_tilde_dc(t_exciter_juicy_tilde* x){ x->click_is_dc=1; }
static void exciter_juicy_tilde_filtermode(t_exciter_juicy_tilde* x, t_symbol* s, int argc, t_atom* argv){
    (void)s; if(argc<1) return; t_symbol* m = atom_getsymbolarg(0,argc,argv);
    if(m==gensym("lp")||m==gensym("lowpass")) x->filter_mode=0;
    else if(m==gensym("hp")||m==gensym("highpass")) x->filter_mode=1;
    else if(m==gensym("bp")||m==gensym("bandpass")) x->filter_mode=2;
}

// new/free
static void* exciter_juicy_tilde_new(void){
    t_exciter_juicy_tilde* x = (t_exciter_juicy_tilde*)pd_new(exciter_juicy_tilde_class);
    if(!x) return NULL;
    x->x_outL = outlet_new(&x->x_obj, &s_signal);
    x->x_outR = outlet_new(&x->x_obj, &s_signal);
    x->sr = sys_getsr(); if(x->sr<=0) x->sr=48000.f;
    x->atk_ms_t=0.f; x->dec_ms_t=40.f; x->sus_t=0.6f; x->rel_ms_t=80.f;
    x->click_vol_t=0.05f; x->noise_gain_t=0.8f; x->filter_norm_t=0.95f; x->hardness_t=0.4f;
    smooth_init(&x->click_vol_s,x->click_vol_t);
    smooth_init(&x->noise_gain_s,x->noise_gain_t);
    smooth_init(&x->filter_norm_s,x->filter_norm_t);
    smooth_init(&x->hardness_s,x->hardness_t);
    smooth_init(&x->dc_s,0.f);
    rng32_seed(&x->rngL,2222u); rng32_seed(&x->rngR,7777u);
    svf_init(&x->svfL); svf_init(&x->svfR); adsr_init(&x->env,x->sr);
    x->filter_mode=0; x->click_is_dc=0; x->gate=0; x->velocity01=1.f;
    x->pending_impulse=0.f; x->dc_target=0.f; x->n_lpL = x->n_lpR = 0.f; x->n_lp_a = 0.0f;
    // 8 float inlets
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
static void exciter_juicy_tilde_free(t_exciter_juicy_tilde* x){ (void)x; }

void exciter_juicy_tilde_setup(void){
    t_class* c = class_new(gensym("exciter_juicy~"),
        (t_newmethod)exciter_juicy_tilde_new,
        (t_method)exciter_juicy_tilde_free,
        sizeof(t_exciter_juicy_tilde),
        CLASS_DEFAULT, 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_dsp, gensym("dsp"), A_CANT, 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_note, gensym("note"), A_GIMME, 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_gate, gensym("gate"), A_FLOAT, 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_vel,  gensym("vel"),  A_FLOAT, 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_impulse, gensym("impulse"), 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_dc,      gensym("dc"), 0);
    class_addmethod(c, (t_method)exciter_juicy_tilde_filtermode, gensym("filter"), A_GIMME, 0);
    CLASS_MAINSIGNALIN(c, t_exciter_juicy_tilde, sr); // visual signal inlet (ignored)
    class_sethelpsymbol(c, gensym("exciter_juicy~-help"));
    exciter_juicy_tilde_class = c;
}
