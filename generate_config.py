# Define the hyperparameters and their ranges
datasets = ["ThreeRing"]
flows = ["chard"]
kernels = ["Gaussian"]
# lmbdas = [1.0, 10.0, 100.0, 1000.0] 
lmbdas = [0.001, 0.0001, 0.00001] # Different values for lambda
step_sizes = [0.01]
bandwidths = [0.15]
step_nums = [500000]
particle_nums = [300]
opts = ["sgd"]
seeds = [42]

# Open a file to write the output
with open('scripts/local_three_ring_chard.sh', 'w') as file:
  file.write("#!/bin/bash\n source activate chard\n which python\n")
  # Generate the command lines
  for dataset in datasets:
    for flow in flows:
      for kernel in kernels:
        for lmbda in lmbdas:
          for step_size in step_sizes:
            for bandwidth in bandwidths:
              for step_num in step_nums:
                for opt in opts:
                  for particle_num in particle_nums:
                    for seed in seeds:
                      command_line = f"python chard/run.py --dataset {dataset} --flow {flow} --kernel {kernel} --lmbda {lmbda} " 
                      command_line += f"--step_size {step_size} --bandwidth {bandwidth} --step_num {step_num} "
                      command_line += f"--source_particle_num {particle_num} --opt {opt} --seed {seed}\n"
                      file.write(command_line)

print("Commands written for chard on Threering dataset!")


# Open a file to write the output
flows = ["mmd"]
bandwidths = [0.3]
step_sizes = [0.01]
with open('scripts/local_three_ring_mmd.sh', 'w') as file:
  # Generate the command lines
  file.write("#!/bin/bash\n source activate chard\n which python\n")
  for dataset in datasets:
    for flow in flows:
      for kernel in kernels:
          for step_size in step_sizes:
            for bandwidth in bandwidths:
              for step_num in step_nums:
                for opt in opts:
                  for particle_num in particle_nums:
                    for seed in seeds:
                      command_line = f"python chard/run.py --dataset {dataset} --flow {flow} --kernel {kernel} " 
                      command_line += f"--step_size {step_size} --bandwidth {bandwidth} --step_num {step_num} "
                      command_line += f"--source_particle_num {particle_num} --opt {opt} --seed {seed}\n"
                      file.write(command_line)

print("Commands written for mmd on Threering dataset")


# Define the hyperparameters and their ranges
device = ["0"]
losss = ["chard"]
lmbdas = [0.01, 0.03, 0.1, 1.0]  # Different values for lambda
lrs = [0.1]
with_noise = ["true"]
noise_decay_freq = ["500"]
seeds = [42]

# Open a file to write the output
with open('scripts/local_student_teacher_config_chard.txt', 'w') as file:
  # Generate the command lines
  for loss in losss:
    for dev in device:
      for lmbda in lmbdas:
        for lr in lrs:
          for noise in with_noise:
            for freq in noise_decay_freq:
              for seed in seeds:
                command_line = f"python student_teacher/train.py --device {dev} --lmbda {lmbda} --loss {loss} " 
                command_line += f"--lr {lr} --with_noise {noise} --noise_decay_freq {freq} --seed {seed}\n"
                file.write(command_line)

print("Commands written for chard on student teacher experiment!")
