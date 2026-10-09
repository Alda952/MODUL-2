import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# 1. DEFINISI SEMESTA DAN FUNGSI KEANGGOTAAN
z = np.linspace(0, 100, 1000)

jelek = fuzz.trapmf(z, [0, 0, 25, 45])
sedang = fuzz.trimf(z, [35, 55, 75])
bagus = fuzz.trapmf(z, [65, 85, 100, 100])

# 2. IMPLIKASI — Pemotongan Kurva (MIN)

alpha_jelek = 0.5
alpha_sedang = 0.8
alpha_bagus = 0.2

imp_jelek = np.fmin(alpha_jelek, jelek)
imp_sedang = np.fmin(alpha_sedang, sedang)
imp_bagus = np.fmin(alpha_bagus, bagus)

# 3. AGREGASI — Penggabungan (MAX)
agregasi = np.fmax(imp_jelek, np.fmax(imp_sedang, imp_bagus))

# 4. DEFUZZIFIKASI — 5 Metode
metode = ['centroid', 'bisector', 'mom', 'som', 'lom']
hasil = {}

for m in metode:
    hasil[m] = fuzz.defuzz(z, agregasi, m)

# 5. TAMPILKAN HASIL
print("HASIL KOMPARASI 5 METODE DEFUZZIFIKASI")
print("Kasus: Kualitas Sinyal WiFi Kampus")
print("=" * 60)
for nama, nilai in hasil.items():
    print(f"• {nama.upper():<10} : {nilai:.3f}")
print("=" * 60)

# 6. VISUALISASI HASIL
plt.figure(figsize=(10, 6))
plt.plot(z, agregasi, 'b-', linewidth=2.5, label='Kurva Agregasi Mamdani')
plt.fill_between(z, 0, agregasi, color='lightblue', alpha=0.4)

warna = {
    'centroid': 'red',
    'bisector': 'green',
    'mom': 'orange',
    'som': 'purple',
    'lom': 'brown'
}

for m, val in hasil.items():
    plt.axvline(x=val, color=warna[m], linestyle='--', linewidth=2,
                label=f'{m.upper()} = {val:.2f}')

plt.title('Perbandingan 5 Metode Defuzzifikasi — Kualitas Sinyal WiFi',
          fontsize=13, fontweight='bold')
plt.xlabel('Kualitas Sinyal', fontsize=11)
plt.ylabel('Derajat Keanggotaan μ(z)', fontsize=11)
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('komparasi_metode_defuzzifikasi.png', dpi=300)
plt.show()
