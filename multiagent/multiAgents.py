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
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # personal code below
        oldGhostPositions = currentGameState.getGhostPositions()
        newGhostPositions = successorGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()
        
        # print("current score: ", currentGameState.getScore())
        # eval_score = successorGameState.getScore()
        eval_score = 0
        oldPos = currentGameState.getPacmanPosition()
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            #       - food will be eaten
            print("food eaten")
            eval_score += 2500

        if (currentGameState.getNumAgents() > successorGameState.getNumAgents()):
            #       - # of ghosts decreases
            # print("ghost killed")
            eval_score += 1000
        
        if (currentGameState.hasFood(newPos[0], newPos[1])):
            eval_score += 1000

        def isClosestFood(state, destination): # pulled from my proj 1
            foodGrid = state.getFood() # can iterate through with asList
            position = state.getPacmanPosition()
            closestPosition = position
            closestDistance = 0
            x = 0
            y = 0
            for row in foodGrid:
                # print(foodGrid)
                for entry in row:
                    if (entry):
                        distance = manhattanDistance((x, y), position) # fails consistency test
                        # distance = mazeDistance((x, y), farthestPosition, problem.startingGameState)
                        if (distance < closestDistance):
                            # distance = mazeDistance((x, y), farthestPosition, problem.startingGameState)
                            closestPosition = (x, y)
                            closestDistance = distance
                    y += 1
                x += 1
                y = 0
            return closestPosition == destination
        

        newPosIsClosest = isClosestFood(successorGameState, newPos)
        if (newPosIsClosest):
            eval_score += 300

        i = 0
        total = 0
        for new_ghost_pos in newGhostPositions:
            new_dist = manhattanDistance(new_ghost_pos, newPos)
            old_dist = manhattanDistance(oldGhostPositions[i], oldPos)
            if (new_dist > old_dist and abs(old_dist - new_dist) <= 5): # distance between ghost and successor > distance between ghost and old state
                if (newScaredTimes[i] > 0):
                    # decrease score if distance increases and ghost can be killed
                    total -= 100
                else:
                    # increase score if distance increases and ghost cannot be killed
                    
                    total += 20 # 50
            elif (new_dist < old_dist and abs(old_dist - new_dist) <= 5):
                if (newScaredTimes[i] > 0):
                    # decrease score if distance decreases and ghost can be killed
                    total += 200
                else:
                    # decrease score if distance decreases and ghost cannot be killed
                    # total -= 50
                    total -= 20 # 50
            i += 0

        eval_score += total/len(newGhostPositions)
                
        # 1) increase score if:
        #       - moves pacman closer to a ghost that can be killed, and the remaining time is high
        # 2) decrease score if:
        #       - moves pacman closer to a ghost that can be killed, and the remaining time is low


        # 1) increase score if:
        #       - food will be eaten
        #       - # of ghosts decreases
        #       - moves pacman closer to a ghost that can be killed, and the remaining time is high
        #       - distance to closest food decreases
        # 2) decrease score if:
        #       - distance to ghost decreases and ghost cannot be killed
        #       - moves pacman closer to a ghost that can be killed, and the remaining time is low

        "*** YOUR CODE HERE ***"
        # print("successorGameState: ", successorGameState)
        # print("newPos: ", newPos)
        # print("newFood: ", newFood)
        # print("newGhostStates: ", newGhostStates) # Ghost: (x,y)=(11.0, 4.0), South
        # print("newScaredTimes: ", newScaredTimes)
        print("eval_score: ", eval_score)
        return eval_score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        # pseudocode here
        # get legal actions of pacman
        # could have array of [x, [x, [x]]] --> each entry = a level deeper of actions
        # ghosts want to minimize; pacman wants to maximize
        # depth --> pacman --> ghost 1 --> ghost 2 --> ghost 3 --> [leaf value or pacman] --> ghost 1, etc. 
        # value or action? as return value
        def getActionAndValue(agentIndex: int, action, gameState: GameState, depth: int, totalAgents: int): 
            # gamestate is successor state after applying action with previous agent
            # should return [action, value]
            if (depth == 0 or gameState.isWin() or gameState.isLose()):
                # print("depth 0: ", [action, self.evaluationFunction(gameState)]) 
                # return [action, self.evaluationFunction(gameState)] # Minimax agent determines if its a win + the score 
                # return [action, self.evaluationFunction(gameState.generateSuccessor(agentIndex, move))]
                return [action, self.evaluationFunction(gameState)]
            gameState = gameState.generateSuccessor(agentIndex, action)
            originalAgentIndex = agentIndex
            if (depth == 0 or gameState.isWin() or gameState.isLose()):
                # print("depth 0: ", [action, self.evaluationFunction(gameState)]) 
                # return [action, self.evaluationFunction(gameState)] # Minimax agent determines if its a win + the score 
                # return [action, self.evaluationFunction(gameState.generateSuccessor(agentIndex, move))]
                return [action, self.evaluationFunction(gameState)]
            agentIndex += 1
            if ((agentIndex == totalAgents and totalAgents > 1) or (totalAgents == 1)):
                agentIndex = 0
                depth -= 1

            legalMoves = gameState.getLegalActions(agentIndex)
            nextDepthEntries = list()
            for move in legalMoves:
                # next = [getActionAndValue(agentIndex, move, gameState.generateSuccessor(agentIndex, move), depth, totalAgents)]
                # if (agentIndex == 0 and depth == 0):
                #     newState = gameState.generateSuccessor(agentIndex, move)
                #     next = [action, self.evaluationFunction(newState)]
                # else:
                #     next = getActionAndValue(agentIndex, move, gameState, depth, totalAgents)
                next = getActionAndValue(agentIndex, move, gameState, depth, totalAgents)
                # print(next)
                nextDepthEntries.append(next)

            # print("nextDepthEntries: ", nextDepthEntries)
            if (len(nextDepthEntries) > 0):
                # print("nextDepthEntries: ", nextDepthEntries)
                bestVal = nextDepthEntries[0][1]
                bestAction = nextDepthEntries[0][0]
            else:
                print("Len(nextDepthEntries) == 0")
                print("No action detected?")
                return [None, 0]
            # bestVal = nextDepthEntries[0][1] # length should always be nonzero
            # bestAction = nextDepthEntries[0][0]
            # if (agentIndex == 0):
            if (originalAgentIndex == 0):
                # pacman = maximize
                for act_val_pair in nextDepthEntries:
                    if (act_val_pair[1] > bestVal):
                        bestVal = act_val_pair[1]
                        bestAction = act_val_pair[0]
            else:
                # ghosts = minimize
                # bestVal = float('inf')
                for act_val_pair in nextDepthEntries:
                    if (act_val_pair[1] < bestVal):
                        bestVal = act_val_pair[1]
                        bestAction = act_val_pair[0]
                # need to force recursion if depth > 0
            return [bestAction, bestVal]
        

        # goal: agentIndex: int, action, gameState: GameState, depth: int, totalAgents: int
        # agent index current agent index, action yet to be applied to gameState, gameState, depth, totalAgents
        # increment agent index after passing to next func 
        legalMoves = gameState.getLegalActions()
        # bestPair = [getActionAndValue(0, move, gameState.generateSuccessor(0, move), self.depth, gameState.getNumAgents()) for move in legalMoves]
        bestPair = [getActionAndValue(0, move, gameState, self.depth, gameState.getNumAgents()) for move in legalMoves]
        
        # print(bestPair)
        bestVal = float('-inf') 
        bestAction = None
        for act_val_pair in bestPair:
            if (act_val_pair[1] > bestVal):
                bestVal = act_val_pair[1]
                bestAction = act_val_pair[0]
        # print("Best Action: ", bestAction)
        # print("Best Val: ", bestVal)
        return bestAction
        # current issue: all calls to getActionValue act as though the function works differently







class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
