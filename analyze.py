#!/usr/bin/env python3
"""
Analyse-Script für aufgezeichnete Atemdaten
Verwendung: python analyze_recording.py recordings/20241127_143022
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.signal import welch


def analyze_recording(base_path: str):
    """Lädt und visualisiert eine Aufzeichnung"""
    
    # Daten laden
    raw_df = pd.read_csv(f"{base_path}_raw.csv")
    analysis_df = pd.read_csv(f"{base_path}_analysis.csv")
    signals = np.load(f"{base_path}_signals.npz")
    
    print(f"Aufzeichnung: {base_path}")
    print(f"Dauer: {raw_df['timestamp'].max():.1f}s")
    print(f"Samples: {len(raw_df)}")
    print(f"Durchschnittliche RR: {analysis_df['respiration_rate'].mean():.1f}/min")
    
    # Figure erstellen
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(f"Atemanalyse: {base_path.split('/')[-1]}", fontsize=14)
    
    # 1. Rohsignal
    ax1 = axes[0]
    ax1.plot(raw_df['timestamp'], raw_df['vertical_motion'], 
             'b-', linewidth=0.5, alpha=0.7, label='Rohsignal')
    ax1.set_ylabel('Vertikale Bewegung')
    ax1.set_title('Rohsignal (Optical Flow)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Gefiltertes Signal
    ax2 = axes[1]
    filtered = signals['filtered_signal']
    if len(filtered) > 0:
        # Zeitachse für gefiltertes Signal rekonstruieren
        t_filtered = np.linspace(
            raw_df['timestamp'].max() - len(filtered) / analysis_df['actual_fs'].mean(),
            raw_df['timestamp'].max(),
            len(filtered)
        )
        ax2.plot(t_filtered, filtered, 'g-', linewidth=1, label='Gefiltert (0.1-0.5 Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Gefiltertes Signal (Bandpass)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Atemfrequenz über Zeit
    ax3 = axes[2]
    ax3.plot(analysis_df['timestamp'], analysis_df['respiration_rate'], 
             'r-', linewidth=1.5, label='Atemfrequenz')
    ax3.fill_between(analysis_df['timestamp'], 
                     analysis_df['respiration_rate'] - 2,
                     analysis_df['respiration_rate'] + 2,
                     alpha=0.2, color='red')
    ax3.axhline(y=analysis_df['respiration_rate'].mean(), 
                color='darkred', linestyle='--', label=f"Mittel: {analysis_df['respiration_rate'].mean():.1f}/min")
    ax3.set_ylabel('Atemzüge/min')
    ax3.set_title('Atemfrequenz')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 40])
    
    # 4. Spektrum (PSD)
    ax4 = axes[3]
    if len(filtered) > 10:
        fs = analysis_df['actual_fs'].mean()
        frequencies, psd = welch(filtered, fs=fs, nperseg=min(len(filtered), int(fs * 15)))
        
        # Nur relevanter Bereich
        mask = frequencies <= 1.0
        ax4.semilogy(frequencies[mask] * 60, psd[mask], 'b-', linewidth=1.5)
        ax4.axvline(x=analysis_df['respiration_rate'].mean(), 
                    color='red', linestyle='--', label='Detektierte RR')
        ax4.set_xlabel('Atemfrequenz (Atemzüge/min)')
        ax4.set_ylabel('Power Spectral Density')
        ax4.set_title('Frequenzspektrum')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 60])
    
    plt.tight_layout()
    
    # Speichern
    plot_path = f"{base_path}_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot gespeichert: {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung: python analyze_recording.py <base_path>")
        print("Beispiel: python analyze_recording.py recordings/20241127_143022")
        sys.exit(1)
    
    analyze_recording(sys.argv[1])
