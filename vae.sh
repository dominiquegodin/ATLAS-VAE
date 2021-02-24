# TRAINING AND VALIDATION
python vae.py --n_train=1e6 --n_valid=1e6 --n_test=1e6 --n_W=0 --n_top=1e6 \
              --batch_size=5e3 --n_epochs=500 --lr=1e-4 --beta=1 --n_iter=1 --output_dir=outputs
