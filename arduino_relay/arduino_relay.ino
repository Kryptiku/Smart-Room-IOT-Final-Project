void setup() {
  Serial.begin(9600);
  pinMode(2, OUTPUT); // Relay 1 - Lights
  pinMode(3, OUTPUT); // Relay 2 - Fan

  digitalWrite(2, HIGH); // Start OFF
  digitalWrite(3, HIGH); // Start OFF
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "LIGHT_ON")  digitalWrite(2, LOW);
    if (command == "LIGHT_OFF") digitalWrite(2, HIGH);
    if (command == "FAN_ON")    digitalWrite(3, LOW);
    if (command == "FAN_OFF")   digitalWrite(3, HIGH);
  }
}