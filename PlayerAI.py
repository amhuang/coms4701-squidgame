import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid
from Utils import *
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

    def trapUtility(self, grid: Grid) -> int:

        # find all available moves by Player
        player_moves = grid.get_neighbors(grid.find(self.player_num), only_available = True)
    
        # find all available moves by Opponent
        opp_moves = grid.get_neighbors(grid.find(3 - self.player_num), only_available = True)
    
        return len(player_moves) - (len(opp_moves) * 2)
    
    def getTrapProbability(self, state: Grid, cell: tuple) -> int:

        player_position = state.find(self.player_num)
        manhattan = manhattan_distance(player_position, cell)
        p = 1 - .05 * (manhattan - 1)

        return p

    def minimizeTrap(self, state : Grid, depth : int, cell : tuple, alpha, beta) -> tuple: 

        pp = state.find(self.player_num)
        op = state.find(3 - self.player_num)
                
        # terminal test
        player_moves = len(state.get_neighbors(pp, only_available = True))
        opp_moves = len(state.get_neighbors(op, only_available = True))

        if player_moves == 0 or opp_moves == 0:
            utility = reachable_utility(state, self.player_num)
            return None, utility, cell
        elif depth == 5:
            utility = reachable_utility(state, self.player_num)
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
            utility = reachable_utility(state, self.player_num)
            return None, utility, cell
        elif depth == 5: 
            utility = reachable_utility(state, self.player_num)
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


    def maximizeMove(self, grid, depth, alpha, beta):
        """
        Picks the move that maximizes utility
        Utility = Improved Score (IS) heuristic
        """
        maxChild = grid
        maxUtil = -(sys.maxsize)

        # find all available moves (children)
        pos = grid.find(self.player_num)
        opp_pos = grid.find(3 - self.player_num)
        available_moves = grid.get_neighbors(pos, only_available = True)
        
        # Terminal test
        player_moves = len(available_moves)
        opp_moves = len(grid.get_neighbors(opp_pos, only_available = True))
        
        if depth >= 5 or player_moves == 0 or opp_moves == 0:
            maxUtil = reachable_utility(maxChild, self.player_num)
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
        
        # find good trap throwing options 
        # Currently: considering throws at spaces neighboring Max in the curr state
        opp_pos = grid.find(3 - self.player_num)    # Throwing from
        pos = grid.find(self.player_num)            # Target
        available_neighbors = grid.get_neighbors(pos, only_available = True)
        
        # Terminal test
        player_moves = len(available_neighbors)
        opp_moves = len(grid.get_neighbors(opp_pos, only_available = True))
        if depth >= 5 or player_moves == 0 or opp_moves == 0:
            # Terminate tree, return utiliy
            minUtil = reachable_utility(minChild, self.player_num)
            return(minChild, minUtil)

        # Get state for each trap throw 
        states = [grid.clone().trap(cell) for cell in available_neighbors]
        
        for i, state in enumerate(states):

            # p = probability of success
            _, util = self.maximizeMove(state, depth + 1, alpha, beta)
            p = 1 - 0.05*(manhattan_distance(opp_pos, available_neighbors[i]) - 1)
            util *= p

            if util < minUtil:
                minChild = state
                minUtil = util
            
            if minUtil <= alpha:  # prune (terminate branch)
                minUtil = util
                return (minChild, minUtil)
            if minUtil < beta:
                beta = minUtil

        return (minChild, minUtil)

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
        reachable_utility(grid, self.player_num)
        return new_pos


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

def reachable_dfs(grid: Grid, pos):
    count = 0
    if grid.map[pos] == 0:
        grid.map[pos] = -2
        count += 1

    moves = grid.get_neighbors(pos, only_available = True)
    if len(moves) == 0:
        return count
   
    for move in moves:
        count += reachable_dfs(grid, move)
    
    return count
    
def reachable_utility(grid: Grid, player_num):
    """
    Heuristic: difference between number of reachable squares from
    own position and number of reachable squares from opponent's 
    position
    """
    clone = grid.clone()
    player_moves = reachable_dfs(clone, grid.find(player_num))

    # find all available moves by Opponent
    clone = grid.clone()
    opp_moves = reachable_dfs(clone, grid.find(3 - player_num))
    
    return player_moves - opp_moves
