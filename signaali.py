import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# Määritellään äänen kuuntelu asetukset
FORMAT = pyaudio.paInt16  # Äänen formaatti (16-bittinen)
CHANNELS = 1              # Yksi kanava (mono)
RATE = 44100              # Näytteenottotaajuus (Hz)
CHUNK = 1024              # Näytteen määrä per bufferi (kuuntelee 1024 näytettä kerrallaan)

# Avaa mikrofonin syöttövirta
p = pyaudio.PyAudio()

# Avaa mikrofonilaitteen syöttö
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

print("Mikrofoni kuuntelee... Paina Ctrl+C lopettaaksesi.")

# Valmistellaan graafinen ikkuna taajuuskomponenttien näyttämistä varten
plt.ion()  # Interactive mode
fig, ax = plt.subplots()
x = np.linspace(0, RATE / 2, CHUNK // 2)
line, = ax.plot(x, np.zeros(CHUNK // 2))
ax.set_ylim(0, 50000)  # Voit säätää y-akselin rajoja tarpeen mukaan
ax.set_xlabel('Taajuus (Hz)')
ax.set_ylabel('Amplitudi')

# AD-muunnos raja-arvo
AD_THRESHOLD = 25000

try:
    while True:
        # Lue äänidataa mikrofonilta
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        
        # Suoritetaan FFT
        fft_data = np.fft.fft(audio_data)
        fft_freq = np.fft.fftfreq(len(fft_data), 1 / RATE)
        
        # Poimitaan vain positiiviset taajuudet ja niiden amplitudit
        positive_freq = fft_freq[:CHUNK // 2]
        positive_fft = np.abs(fft_data[:CHUNK // 2])
        
        # AD-muunnos: kaikki arvot yli rajan muutetaan ykkösiksi
        positive_fft = np.where(positive_fft > AD_THRESHOLD, 1, positive_fft)
        
        # Jos amplitudi ylittää raja-arvon, tulostetaan "1"
        if np.any(positive_fft == 1):
            print("1")  # Tulostaa "1" kun joku amplitudi ylittää raja-arvon
        else:
            print("0")

        # Päivitetään graafi
        line.set_ydata(positive_fft)
        plt.draw()
        plt.pause(0.1)  # Päivittää graafin joka 10 ms välein

except KeyboardInterrupt:
    print("\nOhjelma lopetettu.")
    exit()
finally:
    # Sulje äänivirta ja vapauta resursseja
    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.ioff()  # Loppuu interaktiivinen tila
    plt.show()  # Näyttää lopullisen kuvan, jos ohjelma on suljettu
