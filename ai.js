export class MancalaAI {
  constructor(options = {}) {
    // Default options
    this.settings = {
      aiType: 'alpha_beta',  // 'minimax', 'alpha_beta', 'advanced', 'dqn'
      depth: 7  ,
      ...options
    };
  }

  async getMove(game) {
    try {
      const payload = {
        board: game.board,
        ai: this.settings.aiType
      };

      // Only send depth for tree-based algorithms
      if (this.settings.aiType !== 'dqn') {
        payload.depth = this.settings.depth;
      }

      const response = await fetch('http://localhost:5000/ai-move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.move && data.move !== 0) {
        throw new Error('Invalid move received from AI');
      }
      
      return data.move;
    } catch (error) {
      console.error('AI Error:', error);
      // Fallback to random valid move
      const validMoves = this.getValidMoves(game);
      return validMoves.length > 0 
        ? validMoves[Math.floor(Math.random() * validMoves.length)]
        : null;
    }
  }

  getValidMoves(game) {
    // Assuming game.board follows standard 14-element format
    return game.currentPlayer === 0
      ? game.board.slice(0, 6).map((v, i) => v > 0 ? i : -1).filter(i => i >= 0)
      : game.board.slice(7, 13).map((v, i) => v > 0 ? i + 7 : -1).filter(i => i >= 0);
  }
}