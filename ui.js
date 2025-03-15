import { MancalaGame } from './game.js';
import { MancalaAI } from './ai.js';

// ui.js
class MancalaUI {
  constructor() {
    this.game = new MancalaGame();
    this.ai = new MancalaAI();
    this.pits = document.querySelectorAll('.pit');
    this.stores = document.querySelectorAll('.store');
    this.status = document.getElementById('status');
    
    // Add event listeners for AI controls
    document.getElementById('ai-algorithm').addEventListener('change', () => this.updateAISettings());
    document.getElementById('ai-depth').addEventListener('input', () => this.updateAISettings());
    
    // Initialize visibility
    this.toggleDepthVisibility();

    // Add event listeners
    document.getElementById('ai-algorithm').addEventListener('change', () => {
        this.toggleDepthVisibility();
        this.updateAISettings();
    });

    this.initializeEventListeners();
    this.updateUI();
  }

  toggleDepthVisibility() {
    const algorithm = document.getElementById('ai-algorithm').value;
    const depthControl = document.getElementById('depth-control');
    const isTreeSearch = ['alpha_beta', 'advanced', 'minimax'].includes(algorithm);
    
    depthControl.classList.toggle('hidden', !isTreeSearch);
  }

  getAISettings() {
    return {
      aiType: document.getElementById('ai-algorithm').value,
      depth: parseInt(document.getElementById('ai-depth').value)
    };
  }

  updateAISettings() {
    this.ai = new MancalaAI(this.getAISettings());
    console.log('AI Settings Updated:', this.getAISettings());
  }

  initializeEventListeners() {
    this.pits.forEach(pit => {
      pit.addEventListener('click', () => this.handlePitClick(pit));
    });
  }

  handlePitClick(pit) {
    // if (this.game.currentPlayer !== 0) return;
    
    const pitIndex = parseInt(pit.dataset.index);
    console.log(pitIndex);
    if (this.game.makeMove(pitIndex)) {
      this.updateUI();
      if (this.game.gameActive && this.game.currentPlayer === 1) {
        setTimeout(() => this.makeAIMove(), 1000);
      }
    }
  }

  async makeAIMove() {
    const move = await this.ai.getMove(this.game);
    if (move !== null && this.game.makeMove(move)) {
        this.updateUI();
        
        // Check if AI gets another turn
        if (this.game.gameActive && this.game.currentPlayer === 1) {
            setTimeout(() => this.makeAIMove(), 1000);
        }
    }
}

  updateUI() {
    const state = this.game.getGameState();
    
    // Update pits using data-index directly
    this.pits.forEach(pit => {
        const boardIndex = parseInt(pit.dataset.index);
        pit.textContent = state.board[boardIndex];
        pit.classList.toggle('disabled', 
            (this.game.currentPlayer === 0 && boardIndex > 5) ||
            (this.game.currentPlayer === 1 && boardIndex < 7) ||
            state.board[boardIndex] === 0
        );
    });

    // Update stores
    this.stores[0].textContent = state.board[6];
    this.stores[1].textContent = state.board[13];
    
    // Update status
    this.status.textContent = state.gameActive ? 
        `Player ${state.currentPlayer + 1}'s Turn` : 
        this.getWinnerText();
  }

  getWinnerText() {
    const p1 = this.game.board[6];
    const p2 = this.game.board[13];
    
    if (p1 === p2) return "It's a Tie!";
    return `${p1 > p2 ? 'Player 1' : 'AI'} Wins!`;
  }
}

// Initialize the game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => new MancalaUI());