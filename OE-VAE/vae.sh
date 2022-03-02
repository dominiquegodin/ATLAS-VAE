# TRAINING AND VALIDATION
python vae.py --n_train=1e6 --n_valid=1e6 --n_test=1e6  \
              --n_OoD=1e6 --n_sig=1e6                   \
	      --batch_size=5e3 --n_epochs=50            \
              --n_constituents=20                       \
	      --beta=0.1 --lamb=0                       \
              --output_dir=outputs/test                 \
              --plotting=ON --apply_cut=ON              \
              --weight_type=OoD_2d --OE_type=KLD        \
	      #--model_in=model.h5 --scaler_in=scaler.pkl
