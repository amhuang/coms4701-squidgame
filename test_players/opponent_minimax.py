import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid
from Utils import *
import math

class MinimaxAI(BaseAI):

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
        The function should return a tuple of (x,y) coordinates to which the player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        """
        
        """ Moves based on available moves """
        max_grid, max_util = self.maximizeMove(grid, 0, -(sys.maxsize), sys.maxsize)
        new_pos = max_grid.find(self.player_num)
        return new_pos

    def getTrap(self, grid : Grid) -> tuple:
        """ 
        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        """
        
        """PLACEHOLDER FOR SOPHIAS CODE 
        Copied from mediumAI"""
        
        # find players
        opponent = grid.find(3 - self.player_num)

        # find all available cells in the grid
        available_neighbors = grid.get_neighbors(opponent, only_available = True)

        # edge case - if there are no available cell around opponent, then 
        # player constitutes last trap and will win. throwing randomly.
        if not available_neighbors:
            return random.choice(grid.getAvailableCells())
            
        states = [grid.clone().trap(cell) for cell in available_neighbors]

        # find trap that minimizes opponent's moves
        is_scores = np.array([IS(state, 3 - self.player_num) for state in states])

        # throw to one of the available cells randomly
        trap = available_neighbors[np.argmin(is_scores)] 
    
        return trap

    def maximizeMove(self, grid, depth, alpha, beta):
        """
        Picks the move that maximizes utility
        Utility = Improved Score (IS) heuristic
        """
        maxChild = grid
        maxUtil = -(sys.maxsize)

        # find all available moves (children)
        pos = grid.find(self.player_num)
        available_moves = grid.get_neighbors(pos, only_available = True)
        
        if depth >= 5:
            # Terminate tree, return utility
            maxUtil = AIS(maxChild, self.player_num)
            return(maxChild, maxUtil)
        
        # Get state for each available move 
        states = [grid.clone().move(mv, self.player_num) for mv in available_moves]
        
        for state in states:
            _, util = self.minimizeMove(state, depth + 1, alpha, beta)
            
            if util > maxUtil:
                maxChild = state
                maxUtil = util

            if maxUtil >= beta:  # prune (terminate branch)
                maxUtil = util
                return (maxChild, maxUtil)
            if maxUtil > alpha:
                alpha = maxUtil

        return (maxChild, maxUtil)     


    def minimizeMove(self, grid, depth, alpha, beta):
        """
        Finds throw by opponent that minimizes utililty
        Utility = Improved Score (IS) heuristic
        """

        # Initialize minChild to current state
        minChild = grid
        minUtil = sys.maxsize
        opponent = grid.find(3 - self.player_num)

        # find good trap throwing options 
        # Currently: considering throws at spaces neighboring Max in the curr state
        pos = grid.find(self.player_num)
        available_neighbors = grid.get_neighbors(pos, only_available = True)
        
        if depth >= 5:
            # Terminate tree, return utiliy
            minUtil = AIS(minChild, self.player_num)
            return(minChild, minUtil)

        # Get state for each trap throw 
        states = [grid.clone().trap(cell) for cell in available_neighbors]
        
        for state in states:
            _, util = self.maximizeMove(state, depth + 1, alpha, beta)

            if util < minUtil:
                minChild = state
                minUtil = util
            
            if minUtil <= alpha:  # prune (terminate branch)
                minUtil = util
                return (minChild, minUtil)
            if minUtil < beta:
                beta = minUtil

        return (minChild, minUtil)

    def expectiMove(self, neighbors, grid : Grid, intended_position : tuple):
        # find neighboring cells
        neighbors = grid.get_neighbors(intended_position)
        neighbors = [neighbor for neighbor in neighbors if grid.getCellValue(neighbor) <= 0]
        
        n = len(neighbors)
        probs = np.ones(1 + n)
        # compute probability of success, p
        p = 1 - 0.05*(manhattan_distance(player.getPosition(), intended_position) - 1)

        probs[0] = p
        probs[1:] = np.ones(len(neighbors)) * ((1-p)/n)

        # add desired coordinates to neighbors
        neighbors.insert(0, intended_position)



def AM(grid : Grid, player_num):
    ''' get num of available moves '''
    available_moves = grid.get_neighbors(grid.find(player_num), only_available = True)
    return len(available_moves)

def IS(grid : Grid, player_num):
    ''' 
    Improved Score heuristic: the difference between the current 
    number of moves Player (You) can make and the current number of 
    moves the opponent can make
    '''
    # find all available moves by Player
    player_moves = grid.get_neighbors(grid.find(player_num), only_available = True)
    # find all available moves by Opponent
    opp_moves = grid.get_neighbors(grid.find(3 - player_num), only_available = True)
    return len(player_moves) - len(opp_moves)

def AIS(grid : Grid, player_num):
    ''' 
    Aggessive Improved Score heuristic: IS but with a 2:1 ratio
    applied to opp moves
    '''
    # find all available moves by Player
    player_moves = grid.get_neighbors(grid.find(player_num), only_available = True)
    # find all available moves by Opponent
    opp_moves = grid.get_neighbors(grid.find(3 - player_num), only_available = True)
    return len(player_moves) - 2 * len(opp_moves)