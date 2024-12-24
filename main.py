# for combination
import sys
sys.path.append(r"D:\Research\paper2\experiment\baselines\code")
sys.path.append(r"D:\Research\paper2\experiment\baselines\data")
sys.path.append(r"D:\Research\paper2\experiment\baselines\config")

from setting import *
from datasplit import *
from predict import * 
from optimize import *
from train import *

def main():
    # Parameter to call parameter files
    p_set, p_pre, p_opt, p_tr, p_te, p_ds, p_pto = setting()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameter initialization
    data_split = DataSplit(p_set, p_ds, p_opt, device)

    pre = PredictModel(p_pre, p_set)
    solver = OptProblem(p_opt, p_set, device)

    train = Train(p_tr, p_set, p_ds, device)
    opt_train = OptTrain(p_set, p_pto, p_ds, device)
    

    # main processing
    train_dataset, test_dataset = data_split()

    model = pre(train_dataset)
    model = model.to(device)
    print(model)
    if p_set['process_type'] == 'tr':
        check_point = CheckPoint(train_dataset, p_set, device, solver)
    elif p_set['process_type'] == 'te':
        check_point = CheckPoint(test_dataset, p_set, device, solver)

    check_point(model)
    model = train(train_dataset, model, check_point)

    model = opt_train(train_dataset, model, solver, check_point)

if __name__ == "__main__":
    main()