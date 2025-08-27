// exciter_juicy~.c (FIXED DECAY)
// Physical-modelling exciter for Pd/PlugData by Juicy
// ADSR (noise only, ms, velocity-aware), click type (impulse 1-sample or DC while gate on),
// click volume, noise gain, filter (lp/hp/bp with 0–1 → 1Hz..~0.45*sr log map), hardness (anti-sparse).
#include "m_pd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward-declare class pointer early so new() can use it
static t_class* exciter_juicy_tilde_class;

// ---------------- RNG (xorshift32) ----------------
typedef struct { unsigned int state; } rng32_t;
static inline void rng32_seed(rng32_t* r, unsigned int s){ if(!s) s=0xA3C59AC3u; r->state=s; }
static inline float rng32_nextf(rng32_t* r){
    unsigned int x=r->state; x^=x<<13; x^=x>>17; x^=x<<5; r->state=x;
    float u = (x / 4294967296.0f); return 2.f*u - 1.f;
}

// ---------------- State Variable Filter (Chamberlin) -----------
typedef struct {
    float lp, bp, f, q; int mode; // 0 LP, 1 HP, 2 BP
} svf_t;
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
static inline void adsr_set_times(adsr_t* e, float a,float d,float s,float r){
    e->attack_ms=(a<0?0:a); e->decay_ms=(d<0?0:d); e->sustain=(s<0?0:(s>1?1:s)); e->release_ms=(r<0?0:r);
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
            // FIXED: correct comparator so decay takes the specified ms instead of jumping to sustain.
            if( (e->inc>=0.f && e->value>=e->target) || (e->inc<0.f && e->value<=e->target) ){
                e->value=e->target; e->stage=ENV_SUSTAIN; e->inc=0.f;
            } break;
        case ENV_SUSTAIN:
            break;
        case ENV_RELEASE:
            e->value += e->inc;
            if( (e->inc<=0.f && e->value<=e->target) || (e->inc>0.f && e->value>=e->target) ){
                e->value=e->target; e->stage=ENV_IDLE; e->inc=0.f;
            } break;
        default: break;
    }
    return e->value * e->vel_scale;
}

// ---------------- Main object --------------------
typedef struct _exciter_juicy_tilde {
    t_object x_obj; t_outlet* x_out; float sr;
    // targets
    float atk_ms_t, dec_ms_t, sus_t, rel_ms_t;
    float click_vol_t, noise_gain_t, filter_norm_t, hardness_t;
    // smoothed
    smooth_t click_vol_s, noise_gain_s, filter_norm_s, hardness_s;
    // ADSR
    adsr_t env;
    // RNG
    rng32_t rng;
    // filter
    svf_t svf; int filter_mode;
    // click/DC
    int click_is_dc, gate; float velocity01;
    float pending_impulse;
    smooth_t dc_s; float dc_target;
} t_exciter_juicy_tilde;

// helpers
static inline float clamp01(float v){ return v<0.f?0.f:(v>1.f?1.f:v); }
static inline float mixf(float a,float b,float t){ return a + t*(b-a); }

static t_int* exciter_juicy_tilde_perform(t_int* w){
    t_exciter_juicy_tilde* x=(t_exciter_juicy_tilde*)(w[1]);
    int n=(int)(w[2]); t_sample* out=(t_sample*)(w[3]);
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
        svf_set_cut(&x->svf, cutoff, sr);
        x->svf.q=0.707f; x->svf.mode=x->filter_mode;

        // set ADSR times live (ms) + sustain
        x->env.sustain   = clamp01(x->sus_t);
        x->env.attack_ms = (x->atk_ms_t<0?0:x->atk_ms_t);
        x->env.decay_ms  = (x->dec_ms_t<0?0:x->dec_ms_t);
        x->env.release_ms= (x->rel_ms_t<0?0:x->rel_ms_t);
        float env = adsr_tick(&x->env); // vel applied

        // base noise
        float n0 = rng32_nextf(&x->rng);
        // anti-sparse layer
        float folded = 2.f*fabsf(n0)-1.f;
        float squared = (n0>=0.f? n0*n0 : -(n0*n0));
        float anti = 0.5f*folded + 0.5f*squared;
        float noise_mix = mixf(n0, anti, hardness);
        float noise_out = noise_gain * env * noise_mix;

        // click/DC
        float click_dc = 0.f;
        if(x->click_is_dc){
            x->dc_target = (x->gate? click_vol : 0.f);
            click_dc = smooth_tick(&x->dc_s, x->dc_target);
        } else {
            click_dc = x->pending_impulse; x->pending_impulse=0.f;
        }

        float y = svf_process(&x->svf, noise_out + click_dc);
        out[i]=y;
    }
    return (w+4);
}

static void exciter_juicy_tilde_dsp(t_exciter_juicy_tilde* x, t_signal** sp){
    x->sr = sp[0]->s_sr>0? sp[0]->s_sr : 48000.f;
    adsr_init(&x->env, x->sr); x->env.vel_scale = x->velocity01;
    svf_init(&x->svf);
    smooth_set_tau(&x->click_vol_s, 5.f, x->sr);
    smooth_set_tau(&x->noise_gain_s,5.f, x->sr);
    smooth_set_tau(&x->filter_norm_s,10.f,x->sr);
    smooth_set_tau(&x->hardness_s,10.f,x->sr);
    smooth_set_tau(&x->dc_s,5.f,x->sr);
    dsp_add(exciter_juicy_tilde_perform,3,x,sp[0]->s_n,sp[0]->s_vec);
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
    x->x_out = outlet_new(&x->x_obj, &s_signal);
    x->sr = sys_getsr(); if(x->sr<=0) x->sr=48000.f;
    x->atk_ms_t=0.f; x->dec_ms_t=40.f; x->sus_t=0.6f; x->rel_ms_t=80.f;
    x->click_vol_t=0.05f; x->noise_gain_t=0.8f; x->filter_norm_t=0.95f; x->hardness_t=0.4f;
    smooth_init(&x->click_vol_s,x->click_vol_t);
    smooth_init(&x->noise_gain_s,x->noise_gain_t);
    smooth_init(&x->filter_norm_s,x->filter_norm_t);
    smooth_init(&x->hardness_s,x->hardness_t);
    smooth_init(&x->dc_s,0.f);
    rng32_seed(&x->rng,2222u); svf_init(&x->svf); adsr_init(&x->env,x->sr);
    x->filter_mode=0; x->click_is_dc=0; x->gate=0; x->velocity01=1.f;
    x->pending_impulse=0.f; x->dc_target=0.f;
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

// setup
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
