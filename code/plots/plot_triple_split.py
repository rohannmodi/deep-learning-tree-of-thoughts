import matplotlib.pyplot as plt
import numpy as np

def create_triple_split_plots():
    plt.style.use('default')
    tasks = ['Game of 24', 'GSM8K', 'MATH']
    
    # --- DATA ---
    acc_tot = [46.7, 100.0, 88.0]
    times = [51.5, 11.9, 75.6]
    calls = [68.5, 6.0, 32.4]
    
    # Common Styling Function
    def style_ax(ax, title):
        ax.set_facecolor('white')
        ax.set_title(title, fontsize=16, fontweight='bold', color='black', pad=15)
        ax.tick_params(colors='black', labelsize=12)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.yaxis.grid(True, color='#eee', linestyle='--')

    # 1. ACCURACY PLOT
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.patch.set_facecolor('white')
    bars1 = ax1.bar(tasks, acc_tot, color='#27ae60', edgecolor='black', width=0.6, zorder=3)
    style_ax(ax1, 'Tree of Thoughts Accuracy')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', color='black')
    ax1.set_ylim(0, 115)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}%', ha='center', color='black', fontweight='bold')
    plt.savefig('plot_accuracy_final.png', dpi=300, bbox_inches='tight')

    # 2. API CALLS PLOT
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.patch.set_facecolor('white')
    bars2 = ax2.bar(tasks, calls, color='#c0392b', edgecolor='black', width=0.6, zorder=3)
    style_ax(ax2, 'API Call Volume')
    ax2.set_ylabel('Average API Calls', fontweight='bold', color='black')
    ax2.set_ylim(0, max(calls) * 1.2)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}', ha='center', color='black', fontweight='bold')
    plt.savefig('plot_calls_final.png', dpi=300, bbox_inches='tight')

    # 3. TIME PLOT
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    fig3.patch.set_facecolor('white')
    bars3 = ax3.bar(tasks, times, color='#d35400', edgecolor='black', width=0.6, zorder=3)
    style_ax(ax3, 'Inference Latency (Avg)')
    ax3.set_ylabel('Time (Seconds)', fontweight='bold', color='black')
    ax3.set_ylim(0, max(times) * 1.2)
    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}s', ha='center', color='black', fontweight='bold')
    plt.savefig('plot_time_final.png', dpi=300, bbox_inches='tight')
    
    print("Triple split plots saved.")

if __name__ == '__main__':
    create_triple_split_plots()
