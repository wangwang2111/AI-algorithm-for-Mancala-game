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
    const pitIndex = parseInt(pit.dataset.index);
    if (this.game.makeMove(pitIndex)) {
        pit.classList.add('clicked'); // Add clicked class for visual feedback
        setTimeout(() => {
            pit.classList.remove('clicked'); // Remove clicked class after animation
        }, 500); // Match the duration of the CSS transition
        this.animateStones(pitIndex);
        this.updateUI();
        if (this.game.gameActive && this.game.currentPlayer === 1) {
            setTimeout(() => this.makeAIMove(), 1000);
        }
    }
  }

  animateStones(pitIndex) {
    this.playStoneSound(); // Play sound effect
    const pit = this.pits[pitIndex];
    const stones = this.game.board[pitIndex];
    const stoneElements = [];

    // Create stone elements
    for (let i = 0; i < stones; i++) {
        const stone = document.createElement('div');
        stone.classList.add('stone', 'animated'); // Add 'animated' class
        pit.appendChild(stone);
        stoneElements.push(stone);
    }

    // Animate stones moving to the next pits
    let currentIndex = pitIndex;
    stoneElements.forEach((stone, index) => {
        setTimeout(() => {
            currentIndex = (currentIndex + 1) % 14;
            const nextPit = this.pits[currentIndex];
            const rect = nextPit.getBoundingClientRect();
            stone.style.position = 'absolute';
            stone.style.left = `${rect.left - pit.getBoundingClientRect().left}px`;
            stone.style.top = `${rect.top - pit.getBoundingClientRect().top}px`;
            stone.style.transition = 'all 1s ease';
        }, index * 1000); // Delay each stone's animation
    });

    // Remove stones after animation
    setTimeout(() => {
        stoneElements.forEach(stone => stone.remove());
    }, stones * 1000 + 500);
  }

  async makeAIMove() {
    this.showLoadingSpinner();
    const move = await this.ai.getMove(this.game);
    this.hideLoadingSpinner();

    if (move !== null && this.game.makeMove(move)) {
        this.updateUI();
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
        pit.textContent = ''; // Clear previous stones
        for (let i = 0; i < state.board[boardIndex]; i++) {
            const stone = document.createElement('div');
            stone.classList.add('stone');
            pit.appendChild(stone);
        }
        pit.classList.toggle('disabled', 
            (this.game.currentPlayer === 0 && boardIndex > 5) ||
            (this.game.currentPlayer === 1 && boardIndex < 7) ||
            state.board[boardIndex] === 0
        );
    });

    // Update pit numbers based on data-index
    const pitNumbers = document.querySelectorAll('.pit-number');
    pitNumbers.forEach((pitNumber) => {
        const pitContainer = pitNumber.closest('.pit-container'); // Get the parent pit-container
        const pit = pitContainer.querySelector('.pit'); // Get the pit element inside the container
        const pitIndex = parseInt(pit.dataset.index); // Get the data-index of the pit
        pitNumber.textContent = state.board[pitIndex]; // Update the pit number based on the correct index
    });

    // Update stores with stones instead of numbers
    this.stores.forEach((store, index) => {
        const storeIndex = index === 0 ? 6 : 13; // Player 1 store is at index 6, Player 2 at index 13
        store.textContent = ''; // Clear previous stones
        for (let i = 0; i < state.board[storeIndex]; i++) {
            const stone = document.createElement('div');
            stone.classList.add('stone');
            store.appendChild(stone);
        }
    });

    // Update store numbers (optional, if you still want to show the count)
    const storeCounts = document.querySelectorAll('.store-count');
    storeCounts[0].textContent = state.board[6]; // Player 1 store
    storeCounts[1].textContent = state.board[13]; // Player 2 store

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

  showLoadingSpinner() {
    const spinner = document.getElementById('loading-spinner');
    spinner.classList.remove('hidden');
  }

  hideLoadingSpinner() {
    const spinner = document.getElementById('loading-spinner');
    spinner.classList.add('hidden');
  }

  playStoneSound() {
    const stoneSound = document.getElementById('stone-sound');
    stoneSound.currentTime = 0;
    stoneSound.play();
  }

  playGameOverSound() {
    const gameOverSound = document.getElementById('game-over-sound');
    gameOverSound.currentTime = 0;
    gameOverSound.play();
  }
}

// Initialize the game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => new MancalaUI());