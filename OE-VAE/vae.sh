# TRAINING AND VALIDATION
python vae.py --n_train=1e6 --n_valid=1e6 --n_test=1e6 --n_W=1e6 --n_top=1e6 --batch_size=5e3 \
              --n_epochs=50 --beta=0.1 --lamb=0 --output_dir=outputs --plotting=ON --lr=1e-3
