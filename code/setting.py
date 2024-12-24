import yaml
config_path = r".\config\\"

def setting(set_file='default'):
    with open(config_path + set_file + '.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
    with open(config_path + 'PredictModel\\' + config['pre_file'] + '.yaml', 'r', encoding='utf-8') as file:
        params_pre = yaml.safe_load(file)

    with open(config_path + 'OptProblem\\' + config['opt_file'] + '.yaml', 'r', encoding='utf-8') as file:
        params_opt = yaml.safe_load(file)
        
    with open(config_path + 'Train\\' + config['tr_file'] + '.yaml', 'r', encoding='utf-8') as file:
        params_tr = yaml.safe_load(file)
    
    with open(config_path + 'Test\\' + config['te_file'] + '.yaml', 'r', encoding='utf-8') as file:
        params_te = yaml.safe_load(file)
        
    with open(config_path + 'DataSplit\\' + config['ds_file'] + '.yaml', 'r', encoding='utf-8') as file:
        params_ds = yaml.safe_load(file)

    with open(config_path + 'PTO\\' + config['pto_file'] + '.yaml', 'r', encoding='utf-8') as file:
        params_pto = yaml.safe_load(file)

    if config['verbose'] == True:
        print("File Config: ", config, '\n')
        print("PredictModel Config: ", params_pre, '\n')
        print("OptProblem Config: ", params_opt, '\n')
        print("Train Config: ", params_tr, '\n')
        print("Test Config: ", params_te, '\n')
        print("DataSplit Config: ", params_ds, '\n')
        print("PTO Config: ", params_pto, '\n')
    
    return config, params_pre, params_opt, params_tr, params_te, params_ds, params_pto