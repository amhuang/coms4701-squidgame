import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid
import math

# TO BE IMPLEMENTED
# 
class PlayerAI(BaseAI):

    def __init__(self) -> None:
        # You may choose to add attributes to your player - up to you!
        super().__init__()
        self.pos = None
        self.player_num = None
    
    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position 

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        """ Returns a random, valid move """
        # find all available moves 
        available_moves = grid.get_neighbors(self.pos, only_available = True)
        # make random move
        new_pos = random.choice(available_moves) if available_moves else None

        return new_pos

    def trapUtility(self, grid: Grid) -> int:

        # find all available moves by Player
        player_moves = grid.get_neighbors(grid.find(self.player_num), only_available = True)
    
        # find all available moves by Opponent
        opp_moves = grid.get_neighbors(grid.find(3 - self.player_num), only_available = True)
    
        return len(player_moves) - (len(opp_moves) * 2)
    
    def getTrapProbability(self, state: Grid, cell: tuple) -> int:

        player_position = state.find(self.player_num)
        manhattan = abs(player_position[0] - cell[0]) + abs(player_position[0] - cell[0])
        p = 1 - .05 * (manhattan - 1)

        return p

    def minimizeTrap(self, state : Grid, depth : int, cell : tuple, alpha, beta) -> tuple: 

        pp = state.find(self.player_num)
        op = state.find(3 - self.player_num)

                
        # terminal test
        player_moves = len(state.get_neighbors(pp, only_available = True))
        opp_moves = len(state.get_neighbors(op, only_available = True))

        if player_moves == 0 or opp_moves == 0:
            utility = self.trapUtility(state)
            return None, utility, cell
        elif depth == 5:
            utility = self.trapUtility(state)
            return None, utility, cell

        alpha = alpha
        beta = beta

        minChild, minUtility, minCell = None, math.inf, None
        available_neighbors = state.get_neighbors(op, only_available = True)
        states = [state.clone().move(cell, 3 - self.player_num) for cell in available_neighbors]

        for i, neighbor in enumerate(states):
            _, cu, cc = self.maximizeTrap(neighbor, depth + 1, available_neighbors[i], alpha, beta)

            if cu < minUtility:
                minChild, minUtility, minCell = neighbor, cu, cc
            
            if minUtility <= alpha:
                break 

            if minUtility < beta:
                beta = minUtility
           

        return minChild, minUtility, minCell


    
    def maximizeTrap(self, state : Grid, depth : int, cell : tuple, alpha, beta) -> tuple:

        pp = state.find(self.player_num)
        op = state.find(3 - self.player_num)

        # terminal test
        player_moves = len(state.get_neighbors(pp, only_available = True))
        opp_moves = len(state.get_neighbors(op, only_available = True))
        
        if opp_moves == 0 or player_moves == 0:
            utility = self.trapUtility(state)
            return None, utility, cell
        elif depth == 5: 
            utility = self.trapUtility(state)
            return None, utility, cell

        maxChild, maxUtility, maxCell = None, -math.inf, None
        all_available = state.getAvailableCells()
        available_neighbors = state.get_neighbors(op, only_available = True)

        alpha = alpha
        beta = beta

        #diagonals
        i = op[0]
        j = op[1]
        while i <= 6 and j <= 6:
            c = [i, j]
            if c in all_available and c not in available_neighbors and c != op and c != pp:
                available_neighbors.append(c)
            i += 1
            j += 1
        i = op[0]
        j = op[1]
        while i <= 6 and j >= 0:
            c = [i, j]
            if c in all_available and c not in available_neighbors and c != op and c != pp:
                available_neighbors.append(c)
            i += 1
            j -= 1
        i = op[0]
        j = op[1]
        while j <= 6 and i >= 0:
            c = [i, j]
            if c in all_available and c not in available_neighbors and c != op and c != pp:
                available_neighbors.append(c)
            j += 1
            i -= 1
        i = op[0]
        j = op[1]
        while i >= 0 and j >= 0:
            c = [i, j]
            if c in all_available and c not in available_neighbors and c != op and c != pp:
                available_neighbors.append(c)
            j -= 1
            i -= 1
        
 
        states = [state.clone().trap(cell) for cell in available_neighbors]
        


        for i, neighbor in enumerate(states):
            _, cu, cc = self.minimizeTrap(neighbor, depth + 1, available_neighbors[i], alpha, beta)
            cu = cu * self.getTrapProbability(state, cc)

            if cu > maxUtility:
                maxChild, maxUtility, maxCell = neighbor, cu, cc
            
            if maxUtility >= beta:
                break 

            if maxUtility > alpha:
                alpha = maxUtility

        return maxChild, maxUtility, maxCell

   
    def getTrap(self, grid : Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        """Get the *intended* trap move of the player"""

        alpha = -math.inf
        beta = math.inf
    
        ms, mu, trap = self.maximizeTrap(grid, 0, None, alpha, beta)

        if trap == None:
            # find all available moves 
            available_moves = grid.getAvailableCells()
            # make random move
            trap = random.choice(available_moves) 
        
    
        return trap





        

    
