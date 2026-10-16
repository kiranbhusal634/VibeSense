import serial
import csv
import time
import os



PORT = "COM6"
BAUDRATE = 921600
DURATION = 30.0

OUTPUT_FILE = "data/raw/adxl345_30sec_raw2.csv"



print("Connecting to ESP32...")

ser = serial.Serial(
    PORT,
    BAUDRATE,
    timeout=1
)

time.sleep(2)
ser.reset_input_buffer()

print("Connected.")



print("Waiting for valid XYZ data...")

while True:
    line = ser.readline().decode(errors="ignore").strip()

    if not line:
        continue

    parts = line.split(",")

    if len(parts) == 3:
        try:
            x = int(parts[0])
            y = int(parts[1])
            z = int(parts[2])

            print("Valid sensor data detected.")
            break

        except ValueError:
            pass



print("\nAcquisition starts in 2 seconds...")
time.sleep(2)

# Tell ESP32 to start if your firmware expects a command
ser.write(b"START\n")

time.sleep(0.1)



print("\n========================================")
print("30-SECOND ACQUISITION STARTED")
print("========================================")

samples = []

start_time = time.perf_counter()
last_report = start_time

while True:

    now = time.perf_counter()

    if now - start_time >= DURATION:
        break

    line = ser.readline().decode(errors="ignore").strip()

    if not line:
        continue

    parts = line.split(",")

    if len(parts) != 3:
        continue

    try:
        x = int(parts[0])
        y = int(parts[1])
        z = int(parts[2])

        samples.append((x, y, z))

    except ValueError:
        continue

    # Progress every second
    if now - last_report >= 1.0:

        elapsed = now - start_time
        rate = len(samples) / elapsed

        print(
            f"Elapsed: {elapsed:5.1f} s | "
            f"Samples: {len(samples):6d} | "
            f"Rate: {rate:7.2f} Hz"
        )

        last_report = now



end_time = time.perf_counter()

actual_duration = end_time - start_time
sample_count = len(samples)
effective_rate = sample_count / actual_duration

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["X", "Y", "Z"])

    writer.writerows(samples)



print("\n========================================")
print("ACQUISITION COMPLETE")
print("========================================")

print(f"Duration:       {actual_duration:.6f} s")
print(f"Samples:        {sample_count}")
print(f"Effective rate: {effective_rate:.3f} Hz")

print("\nFirst 10 samples:")

for sample in samples[:10]:
    print(sample)

print("\nCSV:")
print(os.path.abspath(OUTPUT_FILE))

print("========================================")

ser.close()