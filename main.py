import numpy as np
import matplotlib.pyplot as plt

# Функция для генерации кодов Уолша
def generate_walsh_codes(n):
    if n == 1:
        return np.array([[1]])
    else:
        H = generate_walsh_codes(n // 2)
        return np.block([[H, H], [H, -H]])

# Функция для преобразования строки в двоичный ASCII
def string_to_binary(s):
    return ''.join(format(ord(char), '08b') for char in s)

# Функция для кодирования сообщений с помощью кодов Уолша
def encode_with_walsh(message, walsh_code):
    message_bits = np.array(list(map(int, string_to_binary(message))))
    print(f"Encoding message '{message}' to bits: {message_bits}")
    return walsh_code * message_bits[:, np.newaxis]

# Функция для декодирования полученного сигнала
def decode_signal(received_signal, walsh_codes):
    decoded_words = []
    for code in walsh_codes:
        decoded_signal = np.dot(received_signal, code)
        print(f"Decoded signal projection for code {code}:\n{decoded_signal}")
        if np.sum(decoded_signal) > 0:
            decoded_words.append('1')
        else:
            decoded_words.append('0')
    return decoded_words

# Основная программа
def main():
    # Генерация 8-символьных кодов Уолша
    walsh_codes = generate_walsh_codes(8)
    print("Generated Walsh Codes:")
    print(walsh_codes)

    # Определение слов для передачи
    words = {
        'A': 'GOD',
        'B': 'CAT',
        'C': 'HAM',
        'D': 'SUN'
    }
    
    # Кодирование сообщений с их соответствующими кодами Уолша
    encoded_signals = []
    for i, (key, word) in enumerate(words.items()):
        encoded_signal = encode_with_walsh(word, walsh_codes[i])
        print(f"Encoded signal for station {key}:\n{encoded_signal}")
        encoded_signals.append(encoded_signal)

    # Объединение всех сигналов (сложение закодированных сигналов)
    received_signal = np.sum(encoded_signals, axis=0)
    print(f"Combined received signal:\n{received_signal}")

    # Декодирование полученного сигнала
    decoded_bits = decode_signal(received_signal, walsh_codes)
    print("decoded_bits:", decoded_bits)

    # Визуализация закодированных сигналов для каждой станции
    time = np.arange(encoded_signals[0].shape[0])

    for i, (key, word) in enumerate(words.items()):
        plt.figure(figsize=(12, 4))
        plt.plot(time, encoded_signals[i])
        plt.title(f'Encoded Signal for Station {key} - Message: {word}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.savefig(f"plots/encoded_signal_{key}.png")  # Сохранение графика для каждой станции
        plt.close()  # Закрытие графика

    # Визуализация всех закодированных сигналов и общего сигнала
    plt.figure(figsize=(12, 8))
    
    for i, (key, word) in enumerate(words.items()):
        plt.subplot(5, 1, i + 1)
        plt.plot(time, encoded_signals[i])
        plt.title(f'Encoded Signal for Station {key} - Message: {word}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid()

    plt.subplot(5, 1, 5)
    plt.plot(time, received_signal, color='black')
    plt.title('Combined Received Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/combined_signals.png")  # Сохранение общего графика
    plt.close()  # Закрытие графика

    # Вывод результатов
    print("\n=== Decoded Results ===")
    for idx, key in enumerate(words.keys()):
        print(f"Base Station {key} (Message: '{words[key]}'):")
        if decoded_bits[idx] == '1':
            print("  - Signal received successfully.")
        else:
            print(f"  - Message '{words[key]}' not received.")

        # Дополнительная информация по проекциям
        projection = np.dot(received_signal, walsh_codes[idx])
        projection_value = np.sum(projection)  # Суммируем проекцию для получения скалярного значения
        print(f"  - Projection value: {projection_value} (Interpretation: {'Signal detected' if projection_value > 0 else 'No signal detected'})")

if __name__ == "__main__":
    main()
