import pandas as pd

def extract_dataframes(data):
    """
    Extracts the data from nested data structure based on the channels.
    """
    print("Variables are: ", data.keys())
    print("Channels are:", data["columns"].keys())

    df_dict = {}

    for dataset in data["columns"].keys():
        print(f"Processing datasets: {dataset}")

        for k1, v1 in data["columns"][dataset].items():
            print(f"  {k1}")
            
            for k2, v2 in v1.items():
                print(f"    {k2}")
                
                data_dict = {}
                
                for k3, v3 in v2.items():
                    if hasattr(v3, "value"):  
                        v4 = v3.value
                        
                        if len(v4.shape) == 1:
                            data_dict[k3] = v4
                        else:
                            for i in range(v4.shape[1]):
                                data_dict[f"{k3}_{i+1}"] = v4[:, i]
                
                if data_dict:
                    df = pd.DataFrame(data_dict)  # Create DataFrame
                    df_dict[f"{dataset}_{k1}_{k2}"] = df  # Store DataFrame
    
    return df, df_dict


