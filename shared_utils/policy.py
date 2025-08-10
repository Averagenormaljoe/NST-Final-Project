from tensorflow.keras.mixed_precision import set_global_policy
def control_policy(enable_mixed_precision: bool = False):
    try: 
        if enable_mixed_precision:
            print("Enabled mixed_float16 policy")
            set_global_policy('mixed_float16')
    except Exception as e:
        print(f"Error: {e}")
