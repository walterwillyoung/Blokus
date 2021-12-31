Learn how to play blokus here https://www.ultraboardgames.com/blokus/game-rules.php

Python with NumPy library required

Currently supported is a two player game against a computer
Pieces are named "p" + (number of tiles) + letter, eg. "p3b"
Select a peice you wish to play. The code determines all possible moves for you, you pick which one.
The rudimentary AI choses a move based on reducing the number of playable corners for you, and maximizing the number of playable corner squares for itself, while playing the largest possible peice
Play continues until number of turns equals the number of pieces to start
