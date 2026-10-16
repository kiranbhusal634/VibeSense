#include <SPI.h>

// ============================================================
// ESP32 ↔ ADXL345 PINS
// ============================================================

#define CS_PIN    5
#define SCK_PIN   18
#define MISO_PIN  19
#define MOSI_PIN  23
#define INT1_PIN  27

// ============================================================
// ADXL345 REGISTERS
// ============================================================

#define DEVID        0x00
#define BW_RATE      0x2C
#define POWER_CTL    0x2D
#define INT_ENABLE   0x2E
#define INT_MAP      0x2F
#define INT_SOURCE   0x30
#define DATA_FORMAT  0x31
#define DATAX0       0x32
#define FIFO_CTL     0x38
#define FIFO_STATUS  0x39

// ============================================================
// CONFIGURATION
// ============================================================

#define BAUD_RATE 921600

// ADXL345 nominal 1600 Hz
// Actual rate measured previously ≈ 1559.9 Hz

#define FIFO_WATERMARK 8

SPIClass spi(VSPI);

// ============================================================
// INTERRUPT FLAG
// ============================================================

volatile bool fifoWatermark = false;

void IRAM_ATTR fifoISR()
{
  fifoWatermark = true;
}

// ============================================================
// SPI WRITE
// ============================================================

void writeRegister(uint8_t reg, uint8_t value)
{
  spi.beginTransaction(
    SPISettings(8000000, MSBFIRST, SPI_MODE3)
  );

  digitalWrite(CS_PIN, LOW);

  spi.transfer(reg & 0x3F);
  spi.transfer(value);

  digitalWrite(CS_PIN, HIGH);

  spi.endTransaction();
}

// ============================================================
// SPI READ REGISTER
// ============================================================

uint8_t readRegister(uint8_t reg)
{
  spi.beginTransaction(
    SPISettings(8000000, MSBFIRST, SPI_MODE3)
  );

  digitalWrite(CS_PIN, LOW);

  spi.transfer(0x80 | reg);

  uint8_t value = spi.transfer(0x00);

  digitalWrite(CS_PIN, HIGH);

  spi.endTransaction();

  return value;
}

// ============================================================
// READ ONE XYZ SAMPLE
// ============================================================

void readXYZ(
  int16_t &x,
  int16_t &y,
  int16_t &z
)
{
  uint8_t d[6];

  spi.beginTransaction(
    SPISettings(8000000, MSBFIRST, SPI_MODE3)
  );

  digitalWrite(CS_PIN, LOW);

  // 0xC0 = READ + MULTI-BYTE
  spi.transfer(0xC0 | DATAX0);

  for (int i = 0; i < 6; i++)
  {
    d[i] = spi.transfer(0x00);
  }

  digitalWrite(CS_PIN, HIGH);

  spi.endTransaction();

  x = (int16_t)(
    ((uint16_t)d[1] << 8) | d[0]
  );

  y = (int16_t)(
    ((uint16_t)d[3] << 8) | d[2]
  );

  z = (int16_t)(
    ((uint16_t)d[5] << 8) | d[4]
  );
}

// ============================================================
// DRAIN FIFO
// ============================================================

void drainFIFO()
{
  uint8_t status = readRegister(FIFO_STATUS);

  uint8_t entries = status & 0x3F;

  while (entries > 0)
  {
    int16_t x;
    int16_t y;
    int16_t z;

    readXYZ(x, y, z);

    // ========================================================
    // RAW DATA OUTPUT
    // ========================================================

    Serial.print(x);
    Serial.print(',');
    Serial.print(y);
    Serial.print(',');
    Serial.println(z);

    entries--;
  }
}

// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(BAUD_RATE);

  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  pinMode(INT1_PIN, INPUT);

  spi.begin(
    SCK_PIN,
    MISO_PIN,
    MOSI_PIN,
    CS_PIN
  );

  delay(500);

  // ==========================================================
  // VERIFY ADXL345
  // ==========================================================

  uint8_t id = readRegister(DEVID);

  if (id != 0xE5)
  {
    // Don't print anything except an obvious error.
    Serial.println("ERROR_ADXL345");

    while (1)
    {
      delay(1000);
    }
  }

  // ==========================================================
  // STANDBY
  // ==========================================================

  writeRegister(
    POWER_CTL,
    0x00
  );

  // ==========================================================
  // ±2g FULL RESOLUTION
  // ==========================================================

  writeRegister(
    DATA_FORMAT,
    0x08
  );

  // ==========================================================
  // 1600 Hz NOMINAL ODR
  // ==========================================================

  writeRegister(
    BW_RATE,
    0x0E
  );

  // ==========================================================
  // FIFO STREAM MODE
  //
  // 0x80 = STREAM
  // 0x08 = watermark 8
  //
  // 0x88 = STREAM + WATERMARK 8
  // ==========================================================

  writeRegister(
    FIFO_CTL,
    0x88
  );

  // ==========================================================
  // WATERMARK → INT1
  // ==========================================================

  writeRegister(
    INT_MAP,
    0x00
  );

  // ==========================================================
  // ENABLE WATERMARK INTERRUPT ONLY
  //
  // WATERMARK = bit 1
  // ==========================================================

  writeRegister(
    INT_ENABLE,
    0x02
  );

  // ==========================================================
  // START MEASUREMENT
  // ==========================================================

  writeRegister(
    POWER_CTL,
    0x08
  );

  delay(100);

  // Clear any pending interrupt
  readRegister(INT_SOURCE);

  // ==========================================================
  // ATTACH GPIO27 INTERRUPT
  // ==========================================================

  attachInterrupt(
    digitalPinToInterrupt(INT1_PIN),
    fifoISR,
    RISING
  );

  // ==========================================================
  // CSV HEADER
  // ==========================================================

  Serial.println("X,Y,Z");
}

// ============================================================
// LOOP
// ============================================================

void loop()
{
  if (fifoWatermark)
  {
    fifoWatermark = false;

    drainFIFO();
  }

  // Safety:
  // If the interrupt flag somehow gets missed,
  // check FIFO occasionally.
  //
  // We don't continuously poll it because we want
  // minimal SPI traffic.
  static uint32_t lastSafetyCheck = 0;

  if (micros() - lastSafetyCheck >= 2000)
  {
    lastSafetyCheck = micros();

    uint8_t status =
      readRegister(FIFO_STATUS);

    uint8_t entries =
      status & 0x3F;

    if (entries >= 16)
    {
      drainFIFO();
    }
  }
}