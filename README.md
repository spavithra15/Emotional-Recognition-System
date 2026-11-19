# Emotion Recognition System (Java + OpenCV + ONNX Runtime)

1. Install JDK 17+, Eclipse IDE for Java Developers, and (optionally) Maven.
2. Download the two files listed in `models/README.txt` into `models/`.
3. Eclipse: File > Import > Maven > Existing Maven Projects > select this folder.
4. Right-click `MainApp.java` > Run As > Java Application.
5. Command line alternative: `mvn compile exec:java`
6. Tests: `mvn test`

The working directory must be the project root so `models/...` paths are found.
