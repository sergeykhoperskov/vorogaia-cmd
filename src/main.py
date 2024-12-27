import sys
from tools import file_exists, yes_no_input
from stellar_populations.stellar_population import StellarPopulationModel


def main(config_file_name):

    # Create an instance of the model
    model = StellarPopulationModel()
    
    # Read configuration file
    model.read_config(config_file_name)
    
    # Generate SSP based on IMF
    model.initialize_ssp()
    
    # download Padova isochrones
    model.initialize_isochrones()
    
    # read CMD to fit
    model.create_cmd_grid()
    
    # transfer isochrones to the CMD grid
    model.populate_isochrones()
    
    # make a fit
    # model.fit_cmd()


if __name__ == "__main__":

    args = sys.argv[1:]
    
    if not(len(args) == 1):
        
        print("Usage: python3 main.py <config_file_name>")
        
        if yes_no_input("Do you want to continue with default config.ini ?"):
            config_file_name = 'config.ini'
        else:
            print('Exit now')
            sys.exit(1)            
    else:
        config_file_name = args[0]
        
    if file_exists(config_file_name):
        print('File',config_file_name,'exists')
    else:
        print('No config file',config_file_name,'found')
        sys.exit(1)
                

    # Call the main function
    main(config_file_name)

    sys.exit(1)

