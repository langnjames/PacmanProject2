# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, z
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from cmath import inf
import random

import util
from game import Agent, Directions
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) #outputs a grid of all state locations (score only impacted by time currently)
        newPos = successorGameState.getPacmanPosition() #returns tuple of (x, y) values
        newFood = successorGameState.getFood() #returns grid that has T/F values for where dots are located
        newGhostStates = successorGameState.getGhostStates() #returns object (look into what I could possibly do with this)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #returns 0 for now since there are no capsules to eat/eaten

        "*** YOUR CODE HERE ***"
        
        score = successorGameState.getScore()
        #things that could improve evaluationFunction
        # 1. increased score for being closer to a pellet
        foodDistance = [manhattanDistance (newPos, food) for food in newFood.asList()]
        if foodDistance:
            score += 1 / min(foodDistance)
        # 2. decreased score for being closer to a ghost
        ghostDistance = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        if ghostDistance:
            if min(ghostDistance) < 2:
                score -= 100
            else:
                score += 1 / min(ghostDistance)
        # 3. increased score for being closer to a capsule/ghosts are scared
        

        print("Successor score: ", successorGameState.getScore())
        print("EVAL score: ", score)

        #return statement (don't touch for now) 
        return score


def scoreEvaluationFunction(currentGameState):
    """
    
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Reqs for minimax algorithm:
            # any number of ghosts
            # multiple min layers per max layer (one min layer for each ghost)
            # expand to arbitrary depth
            # score minimax leaves with self.eval
            # single search ply is pacman move and ghost' responses 
                # (in game theory) the number of levels at which branching occurs in a tree of possible outcomes, typically corresponding to the number of moves ahead (in chess strictly half-moves ahead) considered by a computer program.
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState), None
            
            if agentIndex == 0:
                return maxAgent(agentIndex, depth, gameState)
            else: 
                return minAgent(agentIndex, depth, gameState)
        
    
        def maxAgent(agentIndex, depth, gameState):
            maxScore = float(-inf)
            maxAction = None
            for action in gameState.getLegalActions(agentIndex):
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                score, _ = minimax((agentIndex + 1) % gameState.getNumAgents(), depth + 1, nextGameState)
                if score > maxScore:
                    maxScore, maxAction = score, action
            return maxScore, maxAction

        def minAgent(agentIndex, depth, gameState):
            minScore = float(inf)
            minAction = None
            for action in gameState.getLegalActions(agentIndex):
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                score, _ = minimax((agentIndex + 1) % gameState.getNumAgents(), depth + 1, nextGameState)
                if score < minScore:
                    minScore, minAction = score, action
            return minScore, minAction
    
        _ , action = minimax(0, 0, gameState)
        return action

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
