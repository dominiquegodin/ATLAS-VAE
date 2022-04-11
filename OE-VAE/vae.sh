# TRAINING AND VALIDATION
output_dir=outputs #/array_id_${ARRAY_ID}
#output_dir=outputs
#echo $output_dir
#exit

python vae.py --n_train=8e6 --n_valid=1e6 --n_sig=1e6           \
	      --batch_size=1e4 --n_epochs=100 --lr=1e-3         \
	      --beta=1 --lamb=50 --n_const=100                  \
              --OE_type=KLD --weight_type=X-S                   \
              --plotting=ON --apply_cut=ON --scaling=ON         \
              --output_dir=${output_dir} --array_id=${ARRAY_ID} \
              #--scaler_in=scaler.pkl --model_in=model.h5