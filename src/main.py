from datetime import date

from data_preprocessing import preprocess, split_data
from src.features import minimal_feature_engineering
from src.model import train_and_explain, get_optimal_threshold, evaluate


def main(): # pragma: no cover
    data = preprocess(date(2011, 1, 1))
    data = minimal_feature_engineering(data)
    train_X, train_y, test_X, test_y = split_data(data)
    model, probabilities, global_importance_dict, local_attributions = train_and_explain(train_X, train_y)
    threshold = get_optimal_threshold(train_y, probabilities)
    evaluation_dict = evaluate(train_y, probabilities, threshold)
    return 
  

if __name__ == "__main__":  # pragma: no cover
    main()
