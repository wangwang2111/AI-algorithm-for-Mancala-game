export class MancalaGame {
  constructor() {
    // Index map: [P1_0-5, P1_store, P2_0-5, P2_store]
    this.board = Array(14).fill(4);
    this.board[6] = 0;  // Player 1 store
    this.board[13] = 0; // Player 2 store
    this.currentPlayer = 0;
    this.gameActive = true;
}

  makeMove(pitIndex) {
    if (!this.validateMove(pitIndex)) return false;

    let stones = this.board[pitIndex];
    this.board[pitIndex] = 0;
    let currentIndex = pitIndex;

    while (stones > 0) {
      currentIndex = (currentIndex + 1) % 14;

      // Skip opponent's store
      if ((this.currentPlayer === 0 && currentIndex === 13) || 
          (this.currentPlayer === 1 && currentIndex === 6)) {
        continue;
      }

      this.board[currentIndex]++;
      stones--;
    }

    // Capture logic
    this.handleCapture(currentIndex);

    // Check extra turn
    const extraTurn = this.checkExtraTurn(currentIndex);
    if (!extraTurn) {
      this.currentPlayer = 1 - this.currentPlayer;
    }

    this.checkGameEnd();
    return true;
  }

  validateMove(pitIndex) {
    return this.gameActive && 
           this.board[pitIndex] > 0 &&
           ((this.currentPlayer === 0 && pitIndex < 6) ||
           (this.currentPlayer === 1 && pitIndex > 6 && pitIndex < 13));
  }

  handleCapture(currentIndex) {
    if (currentIndex === 6 || currentIndex === 13) return; // Ignore stores
    if (this.board[currentIndex] !== 1) return; // Only capture if the last stone lands in an empty pit

    const isPlayer1Side = this.currentPlayer === 0 && currentIndex < 6;
    const isPlayer2Side = this.currentPlayer === 1 && currentIndex > 6 && currentIndex < 13;

    if (isPlayer1Side || isPlayer2Side) {
      const oppositeIndex = 12 - currentIndex; // Find the opposite pit
      if (this.board[oppositeIndex] > 0) { // Only capture if there are stones in the opposite pit
        const storeIndex = this.currentPlayer === 0 ? 6 : 13;
        this.board[storeIndex] += this.board[oppositeIndex] + 1;
        this.board[oppositeIndex] = 0;
        this.board[currentIndex] = 0;
      }
    }
  }

  checkExtraTurn(currentIndex) {
    return (this.currentPlayer === 0 && currentIndex === 6) ||
           (this.currentPlayer === 1 && currentIndex === 13);
  }

  checkGameEnd() {
    const player1Empty = this.board.slice(0, 6).every(v => v === 0);
    const player2Empty = this.board.slice(7, 13).every(v => v === 0);

    if (player1Empty || player2Empty) {
      this.gameActive = false;

      // Collect remaining stones
      this.board[6] += this.board.slice(0, 6).reduce((a, b) => a + b, 0);
      this.board[13] += this.board.slice(7, 13).reduce((a, b) => a + b, 0);

      // Clear pits
      for (let i = 0; i < 6; i++) this.board[i] = 0;
      for (let i = 7; i < 13; i++) this.board[i] = 0;
    }
  }

  getGameState() {
    return {
      board: [...this.board],
      currentPlayer: this.currentPlayer,
      gameActive: this.gameActive
    };
  }
}
