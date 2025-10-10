import argparse
from correlation import main as corr_analysis
from MI import main as mi_analysis
from decision_tree import main as decision_tree_analysis
from KNN_alldata import main as knn_alldata_analysis
from KNN_resampled import main as knn_resampled_analysis
from DBSCAN import main as dbscan_analysis
import sys

def main():
    parser = argparse.ArgumentParser(description="Run data analysis modules.")
    parser.add_argument(
        "module",
        choices=["correlation", "mi", "decision_tree", "knn_alldata", "knn_resampled", "dbscan", "all"],
        help="The analysis module to run."
    )
    args = parser.parse_args()

    if args.module == "correlation":
        corr_analysis()
    elif args.module == "mi":
        mi_analysis()
    elif args.module == "decision_tree":
        decision_tree_analysis()
    elif args.module == "knn_alldata":
        knn_alldata_analysis()
    elif args.module == "knn_resampled":
        knn_resampled_analysis()
    elif args.module == "dbscan":
        dbscan_analysis()
    elif args.module == "all":
        corr_analysis()
        mi_analysis()
        decision_tree_analysis()
        knn_alldata_analysis()
        knn_resampled_analysis()
        dbscan_analysis()
    else:
        print(f"Unknown module: {args.module}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# To run a specific analysis, use the command line:
# python run.py correlation
# python run.py mi
# python run.py decision_tree
# python run.py knn_alldata
# python run.py knn_resampled
# python run.py dbscan
# python run.py all  # To run all analyses sequentially