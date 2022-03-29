# TRAINING AND VALIDATION
python vae.py --n_train=8e6 --n_valid=1e6               \
              --n_OoD=2e6 --n_sig=1e6 --n_const=20      \
	      --batch_size=1e4 --n_epochs=50 --lr=1e-3  \
	      --beta=2 --lamb=50                        \
              --output_dir=outputs/test                 \
              --weight_type=X-S --OE_type=KLD           \
              --plotting=ON --apply_cut=ON              \
              --scaler_in=scaler.pkl #--model_in=model.h5