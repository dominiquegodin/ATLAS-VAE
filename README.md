# ATLAS-VAE
Variational autoencoder for anomaly detection at the Large Hadron Collider


# Training at LPS (atlas15 or atlas16)
1) Clone framework from GitHub
   ```
   git clone https://github.com/dominiquegodin/ATLAS-VAE.git
   ```
2) Move to framework directory
   ```
   cd ATLAS-VAE
   ```
3) Edit command line in shell script vae.sh
   ```
   emacs vae.sh
   ```
4) Send single job to Slurm manager
   ```
   sbatch -w atlas15 sbatch.sh
   ```
5) Send array jobs to Slurm manager (e.g. for id 1 to 10)
   ```
   sbatch -w atlas15 --array=1-10 sbatch.sh
   ```
6) Report job status
   ```
   squeue
   ```
7) Report NVIDIA GPU status (memory, power usage, temperature, fan speed, etc.)
   ```
   nvidia-smi
   ```
8) Cancel job
   ```
   scancel $job_id
9) Use Slurm interactively and request appropriate ressources
   ```
   ssh atlas15
   cd ATLAS-VAE
   salloc -w atlas15 --time=00:30:00
   . sbatch.sh
   ```
10) Monitor training in real time
   ```
   tail -f -n 1000 $log_file
   ```


# Arguments for vae.py
* n_train: number of training QCD jets
* n_valid: number of validation QCD jets
* n_test: number of test QCD jets
* n_W    : number of training W jets
* n_top  : number of validation top jets
* n_constituents: number of constituents
* batch_size: size of training batches
* n_epochs: number of training epochs
* FCN_neurons: encoder/decoder neurons layout
* latent_dim: latent space dimension
* lr: initial learning rate
* beta: beta parameter
* patience: learning rate patience
* n_iter: number of iterations for validation
* n_gpus: number of GPUs for distributed training
* apply_cut: apply best cut to mass histograms
* plotting: enable plotting performance results
