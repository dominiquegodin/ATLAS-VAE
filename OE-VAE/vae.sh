# TRAINING AND VALIDATION
output_dir=outputs/test
python vae.py --n_train=8e6 --n_valid=1e6 --n_sig=1e6           \
	      --batch_size=1e4 --n_epochs=50 --lr=1e-3          \
	      --beta=2 --lamb=5 --n_const=100                   \
              --OE_type=KLD --weight_type=X-S                   \
              --plotting=ON --apply_cut=ON                      \
              --output_dir=${output_dir} --slurm_id=${SLURM_ID} \
              --scaler_type=QuantileTransformer                 \
              #--scaler_in=QuantileTransformer.pkl               \
              #--model_in=model.h5