Projekt, also Motivation, Ziel, Herangehensweise etc. richtig

# Deep Learning zum Erkennen von handgeschriebenen Buchstaben und Zahlen

Auch wenn die digitalen Medien und Hilfsmittel immer mehr Einzug in unseren Alltag bekommen, ist für die meisten immer noch am natürlichsten etwas mit der Hand zu schreiben. Das lässt sich auch davon ableiten, dass jedes moderne Tablett beispielsweise eine Unterstützung für einen Stift anbietet. Auch wenn das Erkennen von Handschrift keine neue Aufgabe für ein Deep Learning Modell ist, gibt es in diesem Bereich noch einige interessante Details. Modelle werden meistens an riesigen Datensätzen wie MNIST und EMNIST trainiert in denen unzählige Beispiele für handgeschriebenen Buchstaben und Zahlen enthalten sind trainiert. Dabei ist anzumerken, dass diese Beispiele von unterschiedlichen Menschen stammen. Dadurch werden robuste Modelle geschaffen, die ihr Wissen auf die große Varianz der Realität generalisieren können. Sollte jedoch ein Text extrahiert werden, bei der die Person eine sehr individuellen Stil pflegt und die Art und Weise der Schrift stark vom Durchschnitt abweicht, kann es dazu kommen, dass die Performance des Modells nicht ausreicht.

## Projekt

Das Projekt beschäftigt sich mit der Frage ob ein Modell zum Erkennen von Handschrift trainiert werden kann und sich dabei auf die Handschrift eines Individuum spezialisiert. Nach dem Prinzip der "Teachable Machine" könnte die Person das Modell auf seine eigene Handschrift trainieren, um so das Modell von Zeit zu Zeit weiter zu optimieren. Im Rahmen des Projektes wird ein Tool entwickelt, welches die Person auffordert unterschiedliche Buchstaben und Zahlen zu schreiben. Diese Eingaben werden genutzt, um einen gelabelten Datensatz zu erzeugen, der ausschließlich aus den Daten des Nutzers besteht. Auf Grundlage dieser Daten wird ein Deep Learning Modell trainiert und optimiert. Das Modell wird dabei vor die Aufgabe gestellt, bei gegebenem Input in Form eines Bildes den richtigen Buchstaben oder die richtige Zahl vorherzusagen. Damit wäre das ein Modell für eine Categorical Classification. Diese Vorhersagen werden dann mit gängigen Metriken aus dem Bereich der Klassifikation auf Qualität und Performance überprüft.

## Ziel

Ziel des Projektes ist es ein Proof-of-Concept für ein stark individualisiertes und spezialisiertes Deep Learning Modell im Bereich Computer Vision zu entwickeln. Dabei sollen auch die Unterschiede festgestellt werden, wie stark die Performance des Modells abweicht, wenn zusätzliche Datensätze oder nicht verwendet werden. Sollte festgestellt werden, dass auch mit einem kleinen Datensatz, welcher nur vom Nutzer erzeugt wurde, gute Ergebnisse erzielt werden könne, würde das einen weiteren Schritt darstellen, Deep Learning Modelle auch in einer Umgebung zu nutzen, wo keine riesigen Datenmengen vorhanden sind. Vorteil dabei wäre nicht nur die geringere benötigte Datenmenge sondern auch die kürzere Trainingszeit und damit kleinere finanzielle Ressourcen.

## Herangehensweise

Der Ablauf des Projektes ist in 9 Schritte aufgegliedert.

1. Implementierung eines GUI-Tool für die Generierung eines individuellen Datensatzes für unterschiedliche Buchstaben und Zahlen.
2. Sobald das Tool implementiert ist kann bereits angefangen werden Daten für den benötigten Datensatz zu erzeugen. Dieser Datensatz wird von der GUI-Anwendung in strukturierter Form im Arbeitsverzeichnis abgelegt und aufgeteilt.
3. Bevor ein Modell an den Daten trainiert werden kann müssen diese in geeigneter Form vorverarbeitet werden. Dabei werden die Daten über das Python-Modul Tensorflow geladen und umgewandelt.
4. Um ein vollwertiges Training an den Daten durchführen zu können, werden die Daten in ein Trainings-, Validierungs- und Test-Datensatz aufgeteilt.
5. Um den kleinen und individualisierten Datensatz synthetisch zu vergrößern, werden Methoden aus dem Bereich Image-Augmentation in die Vorverarbeitung integriert.
6. Im nächsten Schritt wird das Modell trainiert und mit den gängigen Metriken aus der Image-Classification evaluiert.
7. Der gleiche Vorgang wird noch einmal wiederholt mit dem Unterschied der Integration von zusätzlichen Daten.
8. Am Ende werden beide Modelle miteinander verglichen und die Ergebnisse der Performance gegenübergestellt.
9. Für einen weiteren Feldversuch werden beide Modelle in das bereits zuvor entwickelte GUI-Tool integriert und somit direkt gegen neue Benutzereingaben getestet.

