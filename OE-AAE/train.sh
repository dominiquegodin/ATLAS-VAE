# TRAINING AND VALIDATION
output_dir=outputs/test
python train.py --n_train=1e6  --n_valid=10e6  --n_sig=10e6         \
	        --batch_size=5e3  --n_epochs=0  --lr=1e-3       \
                --beta=1 --lamb=1 --n_const=100                   \
                --weight_type=None  --decorrelation=2d            \
                --plotting=ON --apply_cut=OFF                     \
                --constituents=OFF  --const_scaler_type=QuantileTransformer \
                --HLVs=ON             --HLV_scaler_type=RobustScaler        \
                --output_dir=${output_dir} --slurm_id=${SLURM_ID} \
                --model_in='' --HLV_scaler_in=''                \
                --HLV_scaler_in=HLV_RobustScaler.pkl

                #--const_scaler_in=const_QuantileTransformer.pkl   \
                #--HLV_scaler_in=HLV_RobustScaler.pkl              \
