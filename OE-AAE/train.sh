# TRAINING AND VALIDATION
output_dir=outputs/test
python vae.py --n_train=1e6 --n_valid=1e6 --n_sig=1e6           \
	      --batch_size=1e4 --n_epochs=0 --lr=1e-3          \
	      --beta=2 --lamb=5 --n_const=100                   \
              --OE_type=MAE --weight_type=X-S                   \
              --plotting=ON --apply_cut=OFF                     \
              --decorrelation=2d                                \
              --constituents=OFF  --const_scaler_type=QuantileTransformer \
              --HLVs=ON           --HLV_scaler_type=RobustScaler          \
              --output_dir=${output_dir} --slurm_id=${SLURM_ID} \
              --model_in=model.h5                               \
              --HLV_scaler_in=''

              #--model_in=model.h5                               \
              #--const_scaler_in=const_QuantileTransformer.pkl   \
              #--HLV_scaler_in=HLV_RobustScaler.pkl              \
