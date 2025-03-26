import { MancalaGame } from './components/game.js';
import { MancalaAI } from './components/ai.js';

// Create an audio object globally so it persists across game restarts
const backgroundSound = new Audio('public/background-sound.mp3');
backgroundSound.loop = true; // Ensure it loops continuously
backgroundSound.volume = 0.5; // Adjust volume as needed

function playBackgroundSound() {
  if (backgroundSound.paused) {
    backgroundSound.play().catch(error => console.error("Audio play error:", error));
  }
}

// ui.js
class MancalaUI {
  constructor() {
    this.game = new MancalaGame();
    this.ai = new MancalaAI();
    this.pits = document.querySelectorAll('.pit');
    this.stores = document.querySelectorAll('.store');
    this.status1 = document.getElementById('status1');
    this.status2 = document.getElementById('status2');

    // Track which player the AI controls (0 for Player 1, 1 for Player 2)
    this.aiPlayer = 1; // Start with AI as Player 2
  
    
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

    document.getElementById('restart-button').addEventListener('click', () => {
      this.restartGame();
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
      depth: parseInt(document.getElementById('ai-depth').value),
      currentPlayer: this.aiPlayer === 1 ? 'player_2' : 'player_1'
    };
  }

  updateAISettings() {
    this.ai = new MancalaAI(this.getAISettings());
    console.log('AI Settings Updated:', this.getAISettings());
  }


  initializeEventListeners() {
    this.pits.forEach(pit => {
      pit.addEventListener('click', () => {
        if (!this.animationInProgress) {
          this.handlePitClick(pit);
        }
      });
    });
  }

  async handlePitClick(pit) {
    if (this.animationInProgress) return;

    const pitIndex = parseInt(pit.dataset.index);
    const stones = this.game.board[pitIndex];

    const currentPlayerBeforeMove = this.game.currentPlayer; // Store currentPlayer before move

    if (this.game.makeMove(pitIndex)) {
        this.animationInProgress = true;
        
        // Visual feedback
        pit.classList.add('clicked');
        setTimeout(() => pit.classList.remove('clicked'), 300);

        // Animate stones using the stored currentPlayerBeforeMove
        await this.animateStones(pitIndex, stones, currentPlayerBeforeMove);

        // Update UI after animation
        this.updateUI();
        
        // Reset animation flag BEFORE AI move
        this.animationInProgress = false;
        
        // AI move if needed (now that the flag is cleared)
        if (this.game.gameActive && this.game.currentPlayer === this.aiPlayer) {
          await this.makeAIMove();
        }
      }
    }

  async animateStones(pitIndex, stones, currentPlayerBeforeMove) {
    return new Promise(resolve => {
        this.playStoneSound();
        const pit = Array.from(this.pits).find(p => parseInt(p.dataset.index) === pitIndex);
        if (!pit) {
            console.error(`Pit not found for index ${pitIndex}`);
            resolve();
            return;
        }

        const stoneElements = Array.from(pit.querySelectorAll('.stone'));
        let animationsCompleted = 0;

        stoneElements.forEach((stone, index) => {
            // Start each stone animation sequentially
            setTimeout(async () => {
                stone.classList.add('animated');
                
                // Lift animation
                stone.style.transition = 'transform 0.3s ease-out';
                stone.style.transform = `translate(${Math.random() * 20 - 10}px, ${-20 - Math.random() * 10}px)`;
                
                // Wait for lift to complete
                await new Promise(r => setTimeout(r, 300));
                
                if (currentPlayerBeforeMove === 0 && (pitIndex + index + 1) % 14 === 13) index+=1;
                if (currentPlayerBeforeMove === 1 && (pitIndex + index + 1) % 14 === 6) index+=1;
                // Move to target
                const currentIndex = (pitIndex + index + 1) % 14;
                const targetElement = this.getPitOrStoreElement(currentPlayerBeforeMove, currentIndex);

                if (!targetElement) {
                  console.error("Target element not found!", targetElement, currentIndex,currentPlayerBeforeMove);
                }
                
                if (targetElement) {
                    const targetRect = targetElement.getBoundingClientRect();
                    const sourceRect = pit.getBoundingClientRect();
                    
                    const finalX = targetRect.left - sourceRect.left + (Math.random() * 30 - 15);
                    const finalY = targetRect.top - sourceRect.top + (Math.random() * 30 - 15);
                    
                    stone.style.transition = 'transform 0.5s ease-in-out';
                    stone.style.transform = `translate(${finalX}px, ${finalY}px)`;
                    
                    // Wait for move to complete
                    await new Promise(r => setTimeout(r, 500));
                    
                    targetElement.appendChild(stone);
                    stone.style.transform = '';
                }
                
                animationsCompleted++;
                if (animationsCompleted === stoneElements.length) {
                    resolve();
                }
            }, index * 200); // Stagger animations
        });
    });
  }

  async makeAIMove() {
    if (this.animationInProgress) return;
    this.animationInProgress = true;
    
    const currentPlayerBeforeMove = this.game.currentPlayer;
    try {
        let madeMove;
        do {
            madeMove = false;
            this.showLoadingSpinner();
            
            // Store current state before move
            const preMovePlayer = this.game.currentPlayer;
            
            // In ui.js makeAIMove()
            console.log("Making AI move for player:", this.game.currentPlayer);
          
            const move = await this.ai.getMove(this.game);
            // console.log("Current AI player: ", this.game.currentPlayer);
            
            if (move !== null) {
                // Make move but don't update UI yet
                madeMove = this.game.makeMove(move);
                
                if (madeMove) {
                    this.hideLoadingSpinner();
                    const stones = this.game.board[move];
                    
                    // Animate using pre-move player state
                    await this.animateStones(move, stones, currentPlayerBeforeMove);
                    
                    // Now update UI with final state
                    this.updateUI();
                    
                    // Special case for consecutive turns
                    if (this.game.currentPlayer === preMovePlayer) {
                        await new Promise(r => setTimeout(r, 1000));
                    }
                }
            }
        } while (madeMove && this.game.gameActive && this.game.currentPlayer === this.aiPlayer);
    } finally {
        this.animationInProgress = false;
    }
  }

  getPitOrStoreElement(currentPlayer, index) {
    if (currentPlayer === 0 && index === 6) return document.getElementById('store1');
    if (currentPlayer === 1 && index === 13) return document.getElementById('store2');
    return Array.from(this.pits).find(p => parseInt(p.dataset.index) === index);
  }

  updateUI() {
    const state = this.game.getGameState();
    console.log(`Updating UI with game state:`, state); // Debugging

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
    this.stores.forEach(store => {
      const storeIndex = parseInt(store.dataset.index); // Player 1 store is at index 6, Player 2 at index 13
      // const storeIndex = index === 0 ? 13 : 6; // Player 1 store is at index 6, Player 2 at index 13
      store.textContent = ''; // Clear previous stones
      for (let i = 0; i < state.board[storeIndex]; i++) {
          const stone = document.createElement('div');
          stone.classList.add('stone');
          store.appendChild(stone);
      }
    });

    // Update store numbers (optional, if you still want to show the count)
    const storeCounts = document.querySelectorAll('.store-count');
    storeCounts[1].textContent = state.board[6]; // Player 1 store
    storeCounts[0].textContent = state.board[13]; // Player 2 store

    // Update status displays
    if (state.gameActive) {
      if (state.currentPlayer === 0) {
          this.status1.textContent = "Player 1's Turn";
          this.status2.textContent = "Player 2";
          this.status1.classList.add('active');
          this.status2.classList.remove('active');
      } else {
          this.status1.textContent = "Player 1";
          this.status2.textContent = "Player 2's Turn";
          this.status1.classList.remove('active');
          this.status2.classList.add('active');
      }
  } else {
      const winnerText = this.getWinnerText();
      this.status1.textContent = winnerText;
      this.status2.textContent = winnerText;
      this.status1.classList.remove('active');
      this.status2.classList.remove('active');

  }
  }

  async restartGame() {
    if (this.animationInProgress) return;
    
    // Toggle AI player between 0 and 1
    this.aiPlayer = 1 - this.aiPlayer;
    console.log(`Game Restarted! AI is now Player ${this.aiPlayer + 1}`);
  
    this.game = new MancalaGame();
    this.updateUI();
    this.updateAISettings();
    this.playRestartSound();
  
    playBackgroundSound();
    this.animationInProgress = false;
    // If AI is Player 1, make its first move after a short delay
    if (this.aiPlayer === 0) {
      // Add a small delay to allow UI to update
      setTimeout(async () => {
        if (this.game.gameActive && this.game.currentPlayer === 0) {
          await this.makeAIMove();
        }
      }, 500);
    }
  }

  playRestartSound() {
    const restartSound = document.getElementById('restart-sound');
    restartSound.currentTime = 0;
    restartSound.play();
  }
  

  getWinnerText() {
    const p1 = this.game.board[6];
    const p2 = this.game.board[13];
    
    if (p1 === p2) return "It's a Tie!";
    if ((this.aiPlayer === 0 && p1 > p2) || (this.aiPlayer === 1 && p1 > p2)) {
      this.playGameOverSound();
    } 
    return `${p1 > p2 ? (this.aiPlayer === 0 ? 'AI' : 'Player 1') : (this.aiPlayer === 1 ? 'AI' : 'Player 2')} Wins!`;
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

// Fix: Allow playing background sound when the user interacts with any game element
document.addEventListener('click', () => {
  playBackgroundSound();
}, { once: true }); // Ensures it runs only once

document.getElementById('mute-button').addEventListener('click', () => {
  backgroundSound.muted = !backgroundSound.muted;
  document.getElementById('mute-button').innerText = backgroundSound.muted ? "🔇 Unmute" : "🔊 Mute";
});

// Initialize the game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => new MancalaUI());
