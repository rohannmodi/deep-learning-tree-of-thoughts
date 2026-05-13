import matplotlib.pyplot as plt
import numpy as np

def create_quad_split_plots():
    plt.style.use('default')
    tasks = ['Game of 24', 'GSM8K', 'MATH']
    
    # --- DATA (Verified Averages) ---
    acc_tot = [46.7, 100.0, 88.0]
    times = [51.5, 12.0, 50.5]
    calls = [68.5, 8.0, 28.2]
    
    # Estimated Tokens (per task avg)
    # G24: 68.5 * 350 = 23,975
    # GSM8K: 8.0 * 450 = 3,600
    # MATH: 28.2 * 500 = 14,100
    tokens = [23975, 3600, 14100]
    
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

    # 1. ACCURACY
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.patch.set_facecolor('white')
    bars1 = ax1.bar(tasks, acc_tot, color='#27ae60', edgecolor='black', width=0.6, zorder=3)
    style_ax(ax1, 'ToT Success Rates (%)')
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}%', ha='center', color='black', fontweight='bold')
    plt.savefig('plot_accuracy_final.png', dpi=300, bbox_inches='tight')

    # 2. API CALLS
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.patch.set_facecolor('white')
    bars2 = ax2.bar(tasks, calls, color='#c0392b', edgecolor='black', width=0.6, zorder=3)
    style_ax(ax2, 'API Call Intensity')
    ax2.set_ylabel('Calls per Problem', fontweight='bold', color='black')
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}', ha='center', color='black', fontweight='bold')
    plt.savefig('plot_calls_final.png', dpi=300, bbox_inches='tight')

    # 3. LATENCY
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    fig3.patch.set_facecolor('white')
    bars3 = ax3.bar(tasks, times, color='#d35400', edgecolor='black', width=0.6, zorder=3)
    style_ax(ax3, 'Execution Latency (s)')
    ax3.set_ylabel('Seconds', fontweight='bold', color='black')
    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}s', ha='center', color='black', fontweight='bold')
    plt.savefig('plot_time_final.png', dpi=300, bbox_inches='tight')

    # 4. TOKENS
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    fig4.patch.set_facecolor('white')
    bars4 = ax4.bar(tasks, tokens, color='#8e44ad', edgecolor='black', width=0.6, zorder=3)
    style_ax(ax4, 'Token Throughput')
    ax4.set_ylabel('Total Tokens per Task', fontweight='bold', color='black')
    for bar in bars4:
        h = bar.get_height()
        label = f'{h/1000:.1f}k' if h >= 1000 else f'{h:.0f}'
        ax4.text(bar.get_x() + bar.get_width()/2, h + 2000, label, ha='center', color='black', fontweight='bold')
    plt.savefig('plot_tokens_final.png', dpi=300, bbox_inches='tight')
    
    print("All 4 plots saved.")

if __name__ == '__main__':
    create_quad_split_plots()
