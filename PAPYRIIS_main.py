"""
This file is for simulating the PAPYRUS + OZIRIIS pipeline (CNN for vZWFS reconstruction)


little description of the CNN problem (central obstruction + sky pupil)
#TODO the problem is that the on sky pupil crops below 90X90 which what CNN requires
    # I made the executive decision to use a 90x90 non-masked pupil with on_sky set to True for OZIRIIS
    # because the on_sky label is necessary
    # and I am using the on sky CNN
    # another problem this introduces is that the pupil is also shifted compared to 1st stage
    # but redoing the 1st stage with the proper pupil is hard so we leave it
    #in conclusion: is_onsky = True + onsky CNN + 0 pupil mask (introducing central obstruction causes divergence after iteration 30)
    # cameras are fixed (for vZWFS bits =None (reduces performance alot and even diverges), FWC = None (overflows))
"""



import numpy as np
from functions import *
import time
from PAPYRIIS_2stage_CNN_RL.OOPAO_PAPYRIIS_env import OOPAO_environment_PAPYRIIS
import os

#----------------------------------------------------Generate 2nd atmosphere#----------------------------------------------------
def name_tag(env):
    return f"r0_{env.atm_2nd.r0:.3f}_V0_{env.atm_2nd.V0:.3f}_L0_{env.atm_2nd.L0:.3f}_tboil_{env.atm_2nd.t_boiling[0]:.3f}"


def atm_gen(savedir, env, sim_length):
    print(f"{name_tag(env)}")
    OPD_screen_1 = env.generate_second_stage_atmosphere(nLoop=sim_length)

    os.makedirs(savedir, exist_ok = True)


    np.savez_compressed(
        f"{savedir}/atm_OPDs_2nd_{name_tag(env)}_multi_layer.npz",
        atm_OPDs_2nd=OPD_screen_1,
        r0=env.atm_2nd.r0,
        L0=env.atm_2nd.L0,
        windSpeed=env.atm_2nd.windSpeed,
        fractionalR0=env.atm_2nd.fractionalR0,
        windDirection=env.atm_2nd.windDirection,
        altitude=env.atm_2nd.altitude
    )


#---------------------------------------------------Project atmosphere to 1st stage#----------------------------------------------------
def proj_1st_stage(loaddir, savedir, env):
    # atm_OPDs_2nd = np.load(f"{loaddir}/atm_OPDs_2nd_{name_tag(env)}_multi_layer.npz")
    # atm_OPDs_2nd = atm_OPDs_2nd["atm_OPDs_2nd"]
    atm_OPDs_2nd = np.load(f"{loaddir}/atm_OPDs_2nd_{name_tag(env)}_multi_layer.npy")
    print(f"atm loaded: _{name_tag(env)}_multi_layer")

    os.makedirs(savedir, exist_ok = True)

    atm_OPDs_1st = env.project_atmosphere_to_first_stage(atm_OPDs_2nd)


    np.savez(f"{savedir}/atm_OPDs_1st_{name_tag(env)}_multi_layer.npz",
            atm_OPDs_1st=atm_OPDs_1st,
            r0=env.atm_2nd.r0,
            L0=env.atm_2nd.L0,
            windSpeed=env.atm_2nd.windSpeed,
            fractionalR0=env.atm_2nd.fractionalR0,
            windDirection=env.atm_2nd.windDirection,
            altitude=env.atm_2nd.altitude
    )



#----------------------------------------------------1st stage CL#----------------------------------------------------

def run_1st_stage(loaddir, savedir, env, sim_length, gainCL):

    atm_OPD_1st = np.load(f"{loaddir}/atm_OPDs_1st_{name_tag(env)}_multi_layer.npz")
    atm_OPD_1st = atm_OPD_1st["atm_OPDs_1st"]
    print(f"atm loaded: _{name_tag(env)}_multi_layer")

    os.makedirs(savedir, exist_ok = True)

    first_stage_results = env.run_first_stage_loop(sim_length, atm_OPD_1st, gainCL = gainCL)


    np.savez(f"{savedir}/results_1st_stage_{name_tag(env)}_multi_layer.npz", **{
        k: v for k, v in first_stage_results.items() 
        if k != "config"
    },
        # Config fields flattened
        nLoop=first_stage_results["config"].nLoop,
        gainCL=first_stage_results["config"].gainCL,
        leak=first_stage_results["config"].leak,
        frame_delay=first_stage_results["config"].frame_delay,
        photon_noise=first_stage_results["config"].photon_noise,
    )

#----------------------------------------------------1st stage residual entering to 2nd (integrator)#----------------------------------------------------
def run_2nd_stage_CL1OL2(loaddir, savedir, env, sim_length, gain_2nd):
    print(f"{loaddir}/results_1st_stage_{name_tag(env)}_multi_layer.npz")
    first_stage_results = np.load(f"{loaddir}/results_1st_stage_{name_tag(env)}_multi_layer.npz")
    residuals_opds_1rst = first_stage_results['residuals_opds_1rst']

    os.makedirs(savedir, exist_ok = True)

    dm_commands = np.zeros((sim_length, 97))
    reconstructed_cmd = np.zeros((sim_length, 97))
    scnd_stage_strehl = np.zeros((sim_length))
    tel_2nd_pupil = 0
    src_opd = np.zeros((sim_length, 90, 90))
    projector_kl_2nd = np.zeros((87, 8100))


    obs, _ = env.reset(residuals_opds_1rst)
    t_start = time.perf_counter()
    for t in range(sim_length):
        action = gain_2nd * obs.unsqueeze(0).unsqueeze(0)
        next_obs, info = env.step(action.squeeze(), residuals_opds_1rst)
        if t%100 == 0:
            t_block = time.perf_counter()
            print(f'iteration {t}, 2nd strehl ratio {info['2nd_stage_strehl']}')

        if (t+1)%100 == 0:
            print(time.perf_counter() - t_start)
        dm_commands[t] =info["dm_commands"]
        reconstructed_cmd[t] =info["reconstructed_cmd"].detach().cpu().numpy()
        scnd_stage_strehl[t] =info["2nd_stage_strehl"]
        src_opd[t] =info["src_opd"]
        tel_2nd_pupil = info["telescope_pupil"]
        projector_kl_2nd = info["projector_kl_2nd"]

        obs = next_obs

    print(time.perf_counter() - t_block)
    np.savez(f"{savedir}/results_2nd_stage_CL1OL2_{name_tag(env)}_multi_layer.npz",
        # Concatenated across iterations
        all_2nd_stage_strehl    = scnd_stage_strehl,
        all_dm_commands         = dm_commands,
        all_reconstructed_cmd   = reconstructed_cmd,
        residual_opds_2nd       = src_opd,
        telescope_pupil         = tel_2nd_pupil,
        projector_kl_2nd        = projector_kl_2nd,
        )

#TODO checklist before running everytime:
    #r_0
    #V_0
    #boiling phi(t+dt) = alpha * phi(t) + sqrt(1 - alpha**2) * phi_noise
        #alpha = exp(-dt / t_boiling)
    #wavelength
    #camera noise



if __name__ == '__main__':
    PAPYRIIS_env = OOPAO_environment_PAPYRIIS()
    savedir_atm = "/home/evinerskas/FATMOSS/generated_atm_2nd_stage_150000"
    savedir_proj = "PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage_150000_fatmoss"
    savedir_1st = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v2_15000/first_stage_gain_05_fatmoss"
    savedir_2nd = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v2_15000/second_stage_gain_0_fatmoss"
    simulation_length  = 150000
    simulation_length_2= 15000
    gain_1st = 0.5 #best was determined to be 0.5
    gain_2nd = 0 #Best was determined to be 0.1

    # dt = 1/400
    # t_boil = 5
    # alpha = np.exp(-dt / t_boil)
    # print(alpha)
    # print(np.sqrt(1 - alpha**2))

    
    # atm_gen(savedir = savedir_atm, env = PAPYRIIS_env, sim_length = simulation_length)
    # proj_1st_stage(loaddir = savedir_atm, savedir = savedir_proj, env = PAPYRIIS_env)
    # run_1st_stage(loaddir = savedir_proj, savedir = savedir_1st, env = PAPYRIIS_env, gainCL = gain_1st, sim_length = simulation_length)
    run_2nd_stage_CL1OL2(loaddir = savedir_1st, savedir = savedir_2nd, env = PAPYRIIS_env, sim_length = simulation_length_2, gain_2nd = gain_2nd)

    #0, 0.1, 0.5, 1, 2, 5
    #1st stage only has 0.1, 0.5 and 5
    #so need 0, 1, 2 (currently running)
    #0 linear failed twice
    #0.1 succeeded
    #0.5 succeeded
    #1 succeeded
    #2 failed once
    

    #fatmoss_400_linear failed despite the integrator working
    #fatmoss_1000 did not fail but performance is very bad since the atm is basically pure noise
    #fatmoss_1000_linear is the same as non-linear, both performed slightly worse than the integrator (which did not perform)


    #TODO PO4AO: linear and non-linear
        #0, 0.1, 0.5, 1, 2, 5
            #PO4AO + linear PO4AO
            #apparently there is a problem how boiling is implemented in OOPAO and you might get a weaker atm with more boiling while 
                #the rest of the settings staying the same
            #there is also a problem that boiling is constant across all modes which it is not irl
            #redo boiling with FATMOSS (go through the math beforehand)

    #TODO your RL run with boiling is still worse than the on sky data
        #after the rerun too...
        #but that would be because for low order modes the boiling (decorrelation) is much longer
            #whereas in OOPAO it is constant across modes (look at Fatmoss?)
    
