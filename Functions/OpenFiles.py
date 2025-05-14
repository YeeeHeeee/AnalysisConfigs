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

def extract_combined_dfs(data, df_dict):
    """
    Automatically extracts all available years and channels from `data`,
    collects the corresponding DataFrames from `df_dict`, combines them per channel,
    and returns a dictionary of combined DataFrames and the number of channels.
    """
    years = list(data.get("datasets_metadata", {}).get("by_datataking_period", {}).keys())
    channels = list(data.get("columns", {}).keys())

    df_per_channel = {}

    for channel in channels:
        dfs = [df_dict.get(f"{channel}_{channel}_{year}_baseline") for year in years]
        dfs = [df for df in dfs if df is not None]

        if dfs:
            df_per_channel[f"df_{channel}"] = pd.concat(dfs, ignore_index=True)
            print(f"Combined {len(dfs)} DataFrame(s) for {channel}")
        else:
            print(f"No data found for {channel}")

    return df_per_channel

