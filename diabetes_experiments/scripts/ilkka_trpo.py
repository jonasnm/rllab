# from rllab.algos.vpg import VPG
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
import sys
# sys.path.insert(0, '/Users/jonas/Documents/git/rllab/diabetes_experiments/')
sys.path.insert(0, '/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/')

from load_and_sim_policy import render_and_plot_policy
from plot_training_progress import plot_training_progress

# File name for saving
RL = 'TRPO'

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn (pip install seaborn) for better plotting!')

<<<<<<< HEAD:diabetes_experiments/scripts/rl_trpo.py
# models = ('HovorkaAbsolute-v0', 'HovorkaBinary-v0', 'HovorkaGaussian-v0', 'HovorkaGaussianInsulin-v0', 'HovorkaHovorka-v0')
# models = ('HovorkaAbsolute-v0', 'HovorkaBinary-v0', 'HovorkaGaussian-v0', 'HovorkaGaussianInsulin-v0')
models = ('HovorkaBinary-v0','HovorkaGaussian-v0')
# models = ('CambridgeAbsolute-v0','CambridgeBinary-v0', 'CambridgeGaussian-v0', 'CambridgeGaussianInsulin-v0')
# models = ('HovorkaRandomAbsolute-v0', 'HovorkaRandomBinary-v0', 'HovorkaRandomGaussian-v0', 'HovorkaRandomGaussianInsulin-v0')
# models = ('HovorkaMealsAbsolute-v0', 'HovorkaMealsBinary-v0', 'HovorkaMealsGaussian-v0', 'HovorkaMealsGaussianInsulin-v0')
# models = ('HovorkaMealsGaussian-v0', 'HovorkaMealsGaussianInsulin-v0')
=======
models = ('HovorkaAbsolute-v0', 'HovorkaBinary-v0', 'HovorkaGaussian-v0', 'HovorkaGaussianInsulin-v0')
# models = ('CambridgeAbsolute-v0','CambridgeBinary-v0', 'CambridgeGaussian-v0', 'CambridgeGaussianInsulin-v0')
>>>>>>> 65d49ac627117efe7528508e47e4ea4f44f47d32:diabetes_experiments/scripts/ilkka_trpo.py

# NN_sizes = ((8,), (32, 32), (100, 50, 25))
# NN_sizes = ((32, 32), (100, 50, 25))
NN_sizes = ((100, 50, 25), (32, 32))

for k in range(len(models)):
    # for i in range(len(NN_sizes)):
    for i in range(1):

        # ==========================================================================
        # OpenAI diabetes envs - HovorkaInterval starts at the same value every time,
        # HovorkaIntervalRandom starts at a random value
        # ==========================================================================

        def run_task(*_):
            env = normalize(GymEnv(models[k]))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            learn_std = True
            init_std = 1

            # hidden_sizes = NN_sizes[i]
            # hidden_sizes=(8,)
            # hidden_sizes=(32, 32)
            hidden_sizes=(100, 50, 25)

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=hidden_sizes,
                learn_std=learn_std,
                init_std=init_std
            )

            # =======================
            # Defining the algorithm
            # =======================
            batch_size = 5000
            n_itr = 200
            gamma = .99
            step_size = 0.01
            # max_path_length = 96,

            # algo = VPG(
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=batch_size,
                # max_path_length=max_path_length,
                n_itr=n_itr,
                discount=gamma,
                step_size=step_size
            )
            algo.train()

        # NN_folder = [str(j) for j in NN_sizes[i]]
        # NN_folder = '_'.join(NN_folder)
        NN_folder = '100_50_25'

<<<<<<< HEAD:diabetes_experiments/scripts/rl_trpo.py
        # log_dir = '/Users/jonas/Dropbox/results/miguel_experiments/cambridge/' + RL + '/' + 'subject_6' + '/' + models[k]
        # log_dir = '/home/jonas/Dropbox/results/attd_final_attempt/' + RL + '/' + models[k]
        log_dir = './'
=======
        log_dir = '/Users/jonas/Dropbox/results/miguel_experiments/hovorkaCGM/' + RL + '/' + models[k]
        # log_dir = './'
>>>>>>> 65d49ac627117efe7528508e47e4ea4f44f47d32:diabetes_experiments/scripts/ilkka_trpo.py
        # Running and saving the experiment
        run_experiment_lite(
            run_task,
            # algo.train(),
            log_dir=log_dir,
            # n_parallel=2,
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            # exp_prefix="Reinforce_" + env_name,
            # exp_prefix=data_dir
            plot=False
        )

        ## Testing the policy
<<<<<<< HEAD:diabetes_experiments/scripts/rl_trpo.py
        # data_dir = '/Users/jonas/Dropbox/results/miguel_experiments/cambridge/' + RL + '/' + 'subject_6' + '/' + models[k] + '/' + RL + '_default'
        # data_dir = '/home/jonas/Dropbox/results/attd_final_attempt/' + RL + '/' + models[k] + '/' + RL + '_default'
        data_dir = './'

        # data_dir = '/Users/jonas/Dropbox/results/miguel_experiments/cambridge/' + RL + '/' + models[k] + '/' + '5000' + '/' + NN_folder + '/' + RL + '_default'
=======
        data_dir = '/Users/jonas/Dropbox/results/miguel_experiments/hovorkaCGM/' + RL + '/' + models[k] + '/' + RL + '_default'
>>>>>>> 65d49ac627117efe7528508e47e4ea4f44f47d32:diabetes_experiments/scripts/ilkka_trpo.py

        filename = log_dir + '/params.pkl'
        figure_filename = data_dir + '.png'
        title = RL + '_' + models[k] + '_' + NN_folder

<<<<<<< HEAD:diabetes_experiments/scripts/rl_trpo.py
        # render_and_plot_policy(filename, figure_filename, title)
=======
        render_and_plot_policy(filename, figure_filename, title)
>>>>>>> 65d49ac627117efe7528508e47e4ea4f44f47d32:diabetes_experiments/scripts/ilkka_trpo.py

        # TODO: Add training progress!
        # plot_training_progress(log_dir + '/progress.csv', data_dir + 'training_progress.png', title + 'training progress')
