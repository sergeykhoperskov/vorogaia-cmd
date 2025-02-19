import sys
import os
import pandas as pd
from tools import file_exists, yes_no_input, copy_file, mkdir
from stellar_populations.stellar_population import StellarPopulationModel

def main(config_file_name, new_params_file):
    model = StellarPopulationModel()
    
    # Read configuration file
    model.read_config(config_file_name)

    df = pd.read_csv(new_params_file)
    
    new_params_name = new_params_file.split(".")[0]

    mkdir(new_params_name)
    os.system('cp '+ config_file_name + ' ' + new_params_name + '/')

    
    for I in range(0,len(df)):
        fnout = new_params_name + '/'+new_params_name+str(I).zfill(4)+'.ini'

        for i in model.parameters.keys():
            for j in model.parameters[i].keys(): 
                if j in df.columns:
                    model.parameters[i][j] = df[j].iloc[I]
        
        model.parameters['Fitting']['model_name'] = new_params_name+str(I).zfill(4)
        
        with open(fnout, "w") as file:  
            for i in model.parameters.keys():
                file.write('['+ str(i) +']\n')
                for j in model.parameters[i].keys():                    
                    file.write(str(j)+'='+model.parameters[i][j]+'\n')
                file.write('\n')
            

if __name__ == "__main__":

    args = sys.argv[1:]

    config_file_name = args[0]
    new_params_file = args[1]
    
    print(config_file_name,new_params_file)
    

    # Call the main function
    main(config_file_name,new_params_file)

    sys.exit(1)
