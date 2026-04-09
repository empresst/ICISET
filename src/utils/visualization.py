# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(Y_test, test_predicts: dict, aa_range=100):
    """Exact plotting style from your notebook"""
    aa = list(range(aa_range))
    plt.figure(figsize=(20, 6))
    plt.plot(aa, Y_test[0][:aa_range], marker='.', label="actual", color='black')
    colors = ['green', 'blue', '#FF337A', 'purple', 'red']
    for i, (name, pred) in enumerate(test_predicts.items()):
        plt.plot(aa, pred[:aa_range], '.-', label=f"{name} prediction", 
                 color=colors[i % len(colors)], linewidth=1.0)
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('Global_active_power', size=14)
    plt.xlabel('Time step', size=14)
    plt.legend(fontsize=16)
    plt.show()