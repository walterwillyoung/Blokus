#Stop weird loop issue where inputs change for some reason


#Make decision framework which counts corners, not moves
#Parallelize

import numpy as np
import copy
from abc import ABC, abstractmethod

class Piece:
    def __init__(self, name, size, grid, isMirrorable, rotations):
        self.grid = grid
        self.isMirrorable = isMirrorable
        self.rotations = rotations
        self.name = name
        self.size = size

class Game:
    def __init__(self, number_of_players, boardx, boardy, numberOfBlocks ):
        self.number_of_players = number_of_players
        self.boardx = boardx
        self.boardy = boardy 
        self.numberOfBlocks = numberOfBlocks
        #Need to define number of players, boardsize, number of pieces, and run createPieces and pieceimport)
        self.boardsize = ( boardx, boardy )
        self.playersPieces = self.pieceimport( number_of_players, boardx, boardy, numberOfBlocks, number_of_players )
        self.pieceArray, self.pieceNameToObj = self.createPieces( numberOfBlocks, number_of_players )
        self.board = np.zeros((boardx,boardy))
        self.turnNumber = 0
    def twoPlayerGame(self):
        #Define the human and CPU players
        human = Human(1,self.pieceArray,self.pieceNameToObj, self.boardsize, self.number_of_players, 4,4)
        
        cpu = CPU(2,self.pieceArray,self.pieceNameToObj, self.boardsize, self.number_of_players, self.boardx-5, self.boardy-5)
        #Loop through to play moves
        gameOver = False
        numberOfPieces = len(self.pieceArray[0])
        while gameOver == False:
            
            self.board = human.makeMove(self.board, self.turnNumber)
            self.board = cpu.makeMove(self.board, self.turnNumber)
            self.turnNumber = self.turnNumber + 1
            if self.turnNumber == numberOfPieces:
                gameOver = True
            print(self.board)
        #Determine winner by adding the size of the pieces left for each player
        humanScore = 0
        cpuScore = 0
        for i in self.pieceArray[human.color-1]:
            pieceCount = self.pieceNameToObj[i].size
            humanScore = humanScore + pieceCount
        for i in self.pieceArray[cpu.color-1]:
            pieceCount = self.pieceNameToObj[i].size
            cpuScore = cpuScore + pieceCount
        print(' Your score:\n',humanScore,'\n','Computer Score:\n',cpuScore)
        if humanScore < cpuScore:
            print('Player wins!')
        if humanScore==cpuScore:
            print('Draw!')
        if humanScore > cpuScore:
            print('Computer wins!')
        
    def getVariables( self ):
        return self.pieceArray
    def createPieces ( self, numberOfBlocks, number_of_players ):
        pc_size = ( 5, 5 )
        
        pc_one = np.zeros( pc_size )
        
        pc_two = np.zeros( pc_size )
        
        pc_threea = np.zeros( pc_size )
        pc_threeb = np.zeros( pc_size )
        
        pc_foura = np.zeros( pc_size )
        pc_fourb = np.zeros( pc_size )
        pc_fourc = np.zeros( pc_size )
        pc_fourd = np.zeros( pc_size )
        pc_foure = np.zeros( pc_size )
        
        pc_fivea = np.zeros( pc_size )
        pc_fiveb = np.zeros( pc_size )
        pc_fivec = np.zeros( pc_size )
        pc_fived = np.zeros( pc_size )
        pc_fivee = np.zeros( pc_size )
        pc_fivef = np.zeros( pc_size )
        pc_fiveg = np.zeros( pc_size )
        pc_fiveh = np.zeros( pc_size )
        pc_fivei = np.zeros( pc_size )
        pc_fivej = np.zeros( pc_size )
        pc_fivek = np.zeros( pc_size )
        pc_fivel = np.zeros( pc_size )
        
    
        #1x1
        pc_one[ 0, 0 ] = 1
        
        #1x2
        pc_two[ 0, 0 ]= 1
        pc_two[ 0, 1 ]= 1
        
        #1x3
        pc_threea[ 0, 0 ]= 1
        pc_threea[ 0, 1 ]= 1
        pc_threea[ 0, 2 ]= 1
        #3 Square L
        pc_threeb[ 0, 0 ]= 1
        pc_threeb[ 0, 1 ]= 1
        pc_threeb[ 1, 0 ]= 1
        
        #1x4
        pc_foura[ 0, 0 ]= 1
        pc_foura[ 0, 1 ]= 1
        pc_foura[ 0, 2 ]= 1
        pc_foura[ 0, 3 ]= 1
        #4 Square L
        pc_fourb[ 0, 0 ]= 1
        pc_fourb[ 0, 1 ]= 1
        pc_fourb[ 0, 2 ]= 1
        pc_fourb[ 1, 2 ]= 1
        #4 Square T
        pc_fourc[ 0, 0 ]= 1
        pc_fourc[ 0, 1 ]= 1
        pc_fourc[ 0, 2 ]= 1
        pc_fourc[ 1, 1 ]= 1
        #4 Square Chair
        pc_fourd[ 0, 0 ]= 1
        pc_fourd[ 0, 1 ]= 1
        pc_fourd[ 1, 1 ]= 1
        pc_fourd[ 1, 2 ]= 1
        #4 Square Square
        pc_foure[ 0, 0 ]= 1
        pc_foure[ 0, 1 ]= 1
        pc_foure[ 1, 0 ]= 1
        pc_foure[ 1, 1 ]= 1
        
        #1x5
        pc_fivea[ 0,0 ]= 1
        pc_fivea[ 0,1 ]= 1
        pc_fivea[ 0,2 ]= 1
        pc_fivea[ 0,3 ]= 1
        pc_fivea[ 0,4 ]= 1
        #5 Square L
        pc_fiveb[ 0,0 ]= 1
        pc_fiveb[ 0,1 ]= 1
        pc_fiveb[ 0,2 ]= 1
        pc_fiveb[ 0,3 ]= 1
        pc_fiveb[ 1,3 ]= 1
        #Zombie
        pc_fivec[ 0,0 ]= 1
        pc_fivec[ 0,1 ]= 1
        pc_fivec[ 0,2 ]= 1
        pc_fivec[ 0,3 ]= 1
        pc_fivec[ 1,2 ]= 1
        #Zombie
        pc_fived[ 0,0 ]= 1
        pc_fived[ 0,1 ]= 1
        pc_fived[ 0,2 ]= 1
        pc_fived[ 1,3 ]= 1
        pc_fived[ 1,2 ]= 1
        #Compact
        pc_fivee[ 0,0 ]= 1
        pc_fivee[ 0,1 ]= 1
        pc_fivee[ 1,0 ]= 1
        pc_fivee[ 1,1 ]= 1
        pc_fivee[ 0,2 ]= 1
        #5 Block T
        pc_fivef[ 0,0 ]= 1
        pc_fivef[ 0,1 ]= 1
        pc_fivef[ 0,2 ]= 1
        pc_fivef[ 1,1 ]= 1
        pc_fivef[ 2,1 ]= 1
        #Weirdo
        pc_fiveg[ 0,0 ]= 1
        pc_fiveg[ 0,1 ]= 1
        pc_fiveg[ 1,1 ]= 1
        pc_fiveg[ 1,2 ]= 1
        pc_fiveg[ 2,1 ]= 1
        #2
        pc_fiveh[ 0,0 ]= 1
        pc_fiveh[ 0,1 ]= 1
        pc_fiveh[ 1,1 ]= 1
        pc_fiveh[ 1,2 ]= 1
        pc_fiveh[ 2,2 ]= 1
        #Stairs
        pc_fivei[ 0,0 ]= 1
        pc_fivei[ 0,1 ]= 1
        pc_fivei[ 1,1 ]= 1
        pc_fivei[ 2,1 ]= 1
        pc_fivei[ 2,2 ]= 1
        #Corner
        pc_fivej[ 0,0 ]= 1
        pc_fivej[ 0,1 ]= 1
        pc_fivej[ 0,2 ]= 1
        pc_fivej[ 1,2 ]= 1
        pc_fivej[ 2,2 ]= 1
        #Jakubik Exchange
        pc_fivek[ 0,0 ]= 1
        pc_fivek[ 0,1 ]= 1
        pc_fivek[ 0,2 ]= 1
        pc_fivek[ 1,2 ]= 1
        pc_fivek[ 1,0 ]= 1
        #Plus
        pc_fivel[ 0,1 ]= 1
        pc_fivel[ 1,1 ]= 1
        pc_fivel[ 2,1 ]= 1
        pc_fivel[ 1,2 ]= 1
        pc_fivel[ 1,0 ]= 1
        
        pieceNameToObj = {}
        pieceArrayTemp = []
        pieceArrayLooped = []
        pieceArray = []
        #Have a list of names pieceArray
        #Have a dictionary that converts these names to Piece objects, pieceNameToObj
        if numberOfBlocks > 0:
            p1 = Piece('p1', 1, pc_one, False, 0)
            pieceNameToObj['p1'] = p1
            pieceArrayTemp.append('p1')
            
        if numberOfBlocks > 1:
            p2 = Piece('p2', 2, pc_two, False, 1)
            pieceNameToObj['p2'] = p2
            pieceArrayTemp.append('p2')
            
        if numberOfBlocks > 2:
            p3a = Piece('p3a', 3, pc_threea, False, 1)
            pieceNameToObj['p3a'] = p3a
            pieceArrayTemp.append('p3a')
            
            p3b = Piece('p3b', 3, pc_threeb, False, 3)
            pieceNameToObj['p3b'] = p3b
            pieceArrayTemp.append('p3b')
            
        if numberOfBlocks > 3:
            p4a = Piece('p4a', 4, pc_foura, False, 1)
            pieceNameToObj['p4a'] = p4a
            pieceArrayTemp.append('p4a')
            
            p4b = Piece('p4b', 4, pc_fourb, True, 3)
            pieceNameToObj['p4b'] = p4b
            pieceArrayTemp.append('p4b')
            
            p4c = Piece('p4c', 4, pc_fourc, False, 3)
            pieceNameToObj['p4c'] = p4c
            pieceArrayTemp.append('p4c')
            
            p4d = Piece('p4d', 4, pc_fourd, True, 1)
            pieceNameToObj['p4d'] = p4d
            pieceArrayTemp.append('p4d')
            
            p4e = Piece('p4e', 4, pc_foure, False, 0)
            pieceNameToObj['p4e'] = p4e
            pieceArrayTemp.append('p4e')
            
        if numberOfBlocks > 4:
            p5a = Piece('p5a', 5, pc_fivea, False, 1)
            pieceNameToObj['p5a'] = p5a
            pieceArrayTemp.append('p5a')
            
            p5b = Piece('p5b', 5, pc_fiveb, True, 3)
            pieceNameToObj['p5b'] = p5b
            pieceArrayTemp.append('p5b')
            
            p5c = Piece('p5c', 5, pc_fivec, True, 3)
            pieceNameToObj['p5c'] = p5c
            pieceArrayTemp.append('p5c')
            
            p5d = Piece('p5d', 5, pc_fived, True, 3)
            pieceNameToObj['p5d'] = p5d
            pieceArrayTemp.append('p5d')
            
            p5e = Piece('p5e', 5, pc_fivee, True, 3)
            pieceNameToObj['p5e'] = p5e
            pieceArrayTemp.append('p5e')
            
            p5f = Piece('p5f', 5, pc_fivef, False, 3)
            pieceNameToObj['p5f'] = p5f
            pieceArrayTemp.append('p5f')
            
            p5g = Piece('p5g', 5, pc_fiveg, True, 3)
            pieceNameToObj['p5g'] = p5g
            pieceArrayTemp.append('p5g')
            
            p5h = Piece('p5h', 5, pc_fiveh, True, 1)
            pieceNameToObj['p5h'] = p5h
            pieceArrayTemp.append('p5h')
            
            p5i = Piece('p5i', 5, pc_fivei, True, 1)
            pieceNameToObj['p5i'] = p5i
            pieceArrayTemp.append('p5i')
            
            p5j = Piece('p5j', 5, pc_fivej, False, 3)
            pieceNameToObj['p5j'] = p5j
            pieceArrayTemp.append('p5j')
            
            p5k = Piece('p5k', 5, pc_fivek, False, 3)
            pieceNameToObj['p5k'] = p5k
            pieceArrayTemp.append('p5k')
            
            p5l = Piece('p5l', 5, pc_fivel, False, 0)
            pieceNameToObj['p5l'] = p5l
            pieceArrayTemp.append('p5l')
        
        #Now fill the piece array with the pieces for each player
        for i in range(number_of_players):
            #Because Python is stupid, I need to make a "deep copy" to avoid having everything link back to pieceArrayTemp
            pieceArrayLooped.append(copy.deepcopy(pieceArrayTemp))
            
        pieceArray = copy.deepcopy(pieceArrayLooped)
        return pieceArray, pieceNameToObj            
    
    def pieceimport( self, players, boardx, boardy, numberOfBlocks, number_of_players ):
        #Create Board
        
        playersPieces = {}
        #Assign all the players their starting pieces as strings
        for player in range(players):
            playersPieces[player], _ = self.createPieces( numberOfBlocks, number_of_players )
    
        return ( playersPieces )

class Player( ABC ):
    
    def __init__(self, color, pieceArray, pieceNameToObj, boardsize, number_of_players, startX, startY):
        self.color = color
        self.pieceArray = pieceArray
        self.pieceNameToObj = pieceNameToObj
        self.boardsize = boardsize
        self.number_of_players = number_of_players
        self.startX = startX
        self.startY = startY
        self.evaluationStep = False
    def overlay ( self, piece, flip, rotation, tx, ty, color ): 
        #Rotates a piece to the specified orientation, then places over the matrix
        #Set the dummy variable, based on input type
        #This allows piece to be input as string or as integer
        pc_placeholder = self.pieceNameToObj[piece].grid

        #Flip the piece along the left-right axis, 0=no, 1=yes
        if flip == 1:
            pc_placeholder = np.fliplr( pc_placeholder )
 
        #Rotate the piece counterclockwise specified number of times
        pc_placeholder = np.rot90( pc_placeholder, int(rotation ))
        
        #Clip matrix into smallest possible rectangle with all occupied squares
        pc_placeholder = np.delete( pc_placeholder , np.where( ~pc_placeholder.any( axis = 1 ) ) [ 0 ] , axis = 0 )
        pc_placeholder = np.delete( pc_placeholder , np.where( ~pc_placeholder.any( axis = 0 ) ) [ 0 ] , axis = 1 )
       
        #Translate piece to the correct place on the board
        #Also checks that the piece is valid based on the shape 
        x = np.size( pc_placeholder , 1 )
        y = np.size( pc_placeholder , 0 )
        valid_onboard = 0
        board_placeholder = np.zeros( self.boardsize )
        if np.shape( board_placeholder[ int(ty) : y + int(ty) , int(tx) : x + int(tx) ] ) == np.shape( pc_placeholder ):
            board_placeholder[ int(ty) : y + int(ty) , int(tx) : x + int(tx) ]  = pc_placeholder
            valid_onboard = 1
        
        #Apply proper dye to make proper color
        board_overlay = board_placeholder * int(color)
        return board_overlay , valid_onboard
        
    def rulecheck ( self, board_overlay, board_input, color ):
        #Check that there are no non-zero values that overlap
        #Input board overlay is just the output of the "overlay" function
        #Initialize variables 0 = nonvalid
        
        valid_edge = 1
        valid_corner = 0
        valid_move = 0
        
        #Create array with values only for the player color, all others are zero
        board_color = ( board_input == color ).astype(int)
        
        #CHECK THAT THE BOARD IS OPEN
        
        #Move is valid if the pairwise product of the two matrices is equal to zero
        
        valid_overlap = np.sum( np.multiply( board_overlay, board_input ) )
    
        #CHECK THAT THE COLOR HAS NO ADJACENT EDGES
        
        #Find any adjacent edges of placed piece and future piece
        #Do this by setting 4 matrices, each shifted by one in each cardinal direction
        #Then use the same procedure as before to detect an overlap
        expanded_overlay = np.zeros( [ self.boardsize[ 0 ] + 2, self.boardsize[ 1 ] +2 ] )
        expanded_overlay[ 1 : self.boardsize[0] + 1 , 1 : self.boardsize[1] + 1 ] = board_overlay
        
        #Initialize shifted matries
        overlay_shifted = np.zeros( [self.boardsize[0], self.boardsize[1], 4] )
        
        #Put values into the matrices with specified shifts in cardinal directions
        overlay_shifted[ :, :, 0 ] = expanded_overlay[ 0 : self.boardsize[ 0 ] , 1 : self.boardsize[ 1 ] + 1 ] 
        overlay_shifted[ :, :, 1 ] = expanded_overlay[ 1 : self.boardsize[ 0 ] + 1 , 0 : self.boardsize[ 1 ] ]
        overlay_shifted[ :, :, 2 ] = expanded_overlay[ 2 : self.boardsize[ 0 ] + 2 , 1 : self.boardsize[ 1 ] + 1 ]
        overlay_shifted[ :, :, 3 ] = expanded_overlay[ 1 : self.boardsize[ 0 ] + 1 , 2 : self.boardsize[ 1 ] + 2 ]      
        
        #Apply the same rules as before to make things match
        overlay_shifted_2D = np.sum( overlay_shifted, 2 )
        #Valid if zero
        valid_edge = np.sum( np.multiply( overlay_shifted_2D, board_color ) )
                    
        #CHECK THAT THE COLOR HAS AN ADJACENT CORNER
        #If this is the first turn, check that the peice occupies the starting square
        if self.turnNumber == 0 and self.evaluationStep == False:
            if board_overlay[self.startX, self.startY] == color:
                valid_corner =1
        else:
            #Find any adjacent edges of placed piece and future piece
            #Do this by setting 4 matrices, each shifted by one diagonally
            #Then use the same procedure as before to detect an overlap
            #Put values into the matrices with specified shifts
            #Repurpose previous matrix
            
            #Initialize the diagonal shifted
            overlay_shifted_diagonal = np.zeros( [self.boardsize[0], self.boardsize[1], 4] )
            
            overlay_shifted_diagonal[ :, :, 0 ] = expanded_overlay[ 0 : self.boardsize[ 0 ] , 0 : self.boardsize[ 1 ] ] 
            overlay_shifted_diagonal[ :, :, 1 ] = expanded_overlay[ 2 : self.boardsize[ 0 ] + 2 , 0 : self.boardsize[ 1 ] ]
            overlay_shifted_diagonal[ :, :, 2 ] = expanded_overlay[ 0 : self.boardsize[ 0 ] , 2 : self.boardsize[ 1 ] + 2 ]
            overlay_shifted_diagonal[ :, :, 3 ] = expanded_overlay[ 2 : self.boardsize[ 0 ] + 2 , 2 : self.boardsize[ 1 ] + 2 ]
            
            overlay_shifted_diagonal_2D = np.sum( overlay_shifted_diagonal, 2 )
            
            #Compare shifted overlay to board, nonzero means valid move
            valid_corner = np.sum( np.multiply( overlay_shifted_diagonal_2D, board_color ) )
        
        #Compile all the valid moves checks to one variable
        if valid_overlap == 0 and valid_edge == 0 and valid_corner != 0:
            valid_move = 1
        return valid_move
    
    def moveenumerator( self, color, board, piece_list, boardsize):
        me_color = color
        me_board = board
        me_move_matrix = 0
        for piece_name in piece_list:
            x1 = self.pieceNameToObj[piece_name] 
            #Iterate through all available pieces
            #Plug into overlay function and see if it is a valid move     
            x2_range = 2 if x1.isMirrorable else 1
            for x2 in range( x2_range ):
                for x3 in range ( x1.rotations + 1 ):
                    for x4 in range( self.boardsize[ 0 ] ):
                        for x5 in range(self.boardsize[ 0 ]):
                            me_move = self.overlay( x1.name , x2, x3, x4, x5, me_color )
                            
                            if me_move [ 1 ] == 1:
                                #If the move was on the board, move to the next step
                                #Check whether the move followed the rules
                                me_rulecheck = self.rulecheck( me_move[ 0 ], me_board, me_color )
                                if me_rulecheck == 1:
                                    #Record the move, if it was valid
                                    
                                    new_move = np.array( [ piece_name, int( x2 ), int( x3 ), int( x4 ), int( x5 ), me_color ] )
                                    #Load the valid move into the matrix
                                    if type( me_move_matrix ) != int:
                                        #Determine if the matrix needs to be initiated
                                        me_move_matrix = np.vstack( ( me_move_matrix, new_move ) )
                                    else:
                                        me_move_matrix = new_move

        return (me_move_matrix)
    @abstractmethod
    def makeMove(self,board,turnNumber):
        pass

class CPU(Player):
    def evaluationfunction( self, color, board, number_of_players ):
        #Define the funciton which evaluates one position
        #Color variable should be the side which has just moved
        #This ensures that the first moves can be valid
        self.evaluationStep = True
        #Define a multiplier to number of corners, and number of edges. Also tell how much to prefer central corners
        cornerWeight = 1
        centralWeight = 1
       
        squaresum = np.zeros( int(number_of_players) )
        cornersCentralWeighted = np.zeros( int(number_of_players) )
        
        for x in range( 1, number_of_players + 1 ):
            #First find number of squares played for you and opponents
            squaresum [ x - 1 ] = np.count_nonzero( board == x )
            #Now find the number of avialable corners. Want corners near center if possible
            #Take advantage of the moveenumerator. All I have to do is find number of squared the p1 can fit
            cornerMoveList = self.moveenumerator( x, board, ['p1'], self.boardsize )
            #I care about the number of moves, and the squares where the move takes place
            if cornerMoveList == 0:
                cornersCentralWeighted[x-1] = 0
            #Make cornerMoveList of the open corners after a given move    
            else:
                
                if cornerMoveList.ndim == 2:
                    cornerSquares = np.array([cornerMoveList[:,3],cornerMoveList[:,4]])
                    cornerSquares = cornerSquares.astype(int)
                    minusCornerSquares = cornerSquares - self.boardsize[0]
                    absCornerSquares = abs(minusCornerSquares)
                    cornerSquaresNormalized = np.minimum(absCornerSquares, cornerSquares)
                    cornerSquaresNormalized = cornerSquaresNormalized + 1
                    cornerSquaresWeighted = np.multiply(cornerSquaresNormalized[0,:],cornerSquaresNormalized[1,:])
                    cornerSquaresWeighted = np.sum(cornerSquaresWeighted)
                    cornersCentralWeighted[ x - 1 ] =  cornerSquaresWeighted
                if cornerMoveList.ndim ==1:
                    cornerSquares = np.array([int(cornerMoveList[3]), int(cornerMoveList[4])])
                    minusCornerSquares = cornerSquares - self.boardsize[0]
                    absCornerSquares = abs(minusCornerSquares)
                    cornerSquaresNormalized = np.minimum(absCornerSquares, cornerSquares)
                    cornerSquaresNormalized = cornerSquaresNormalized + 1
                    cornerSquaresWeighted = np.multiply(cornerSquaresNormalized[0],cornerSquaresNormalized[1])
                    cornerSquaresWeighted = np.sum(cornerSquaresWeighted)
                    cornersCentralWeighted[ x - 1 ] =  cornerSquaresWeighted       
        
        #Initialize the board evaluation function
        board_evaluation = squaresum[color-1]    
        
        #Compare number of corners to weighted average of opponent's moves        
        weight = np.multiply( squaresum, cornersCentralWeighted ) + squaresum

        weight_opponents = weight[ np.arange( len( weight ) ) != color - 1 ]
        weight_self = weight[ color - 1 ]
        
        weight_opponents_average = np.average( weight_opponents )

        if weight_opponents_average != 0:
            board_evaluation = weight_self / weight_opponents_average    
        self.evaluationStep = False
        return board_evaluation
    
    def decisionframework(self, color, board ):
        
        best_move = 0
        #Generate a list of all possible moves usinng the function
        move_list = self.moveenumerator( color, board, self.pieceArray[color-1], self.boardsize )
        #Return if no possible moves
        if move_list == 0:
            return best_move
        #I will store the evaluaiton here of all possible moves
        move_evaluation = np.zeros( np.size( move_list, 0 ) )
        loop = 0
        #Loop through all possible moves and use the evaluation function to judge it
        if move_list.ndim == 1:
            best_move = move_list
            return best_move
    
        counter = 0
        counterMax = np.size(move_list, 0)
        for i in move_list:
            
            print(round(counter/counterMax,2))
            counter = counter + 1
            board_placeholder = self.overlay( i[ 0 ], int(i[ 1 ]), int(i[ 2 ]), int(i[ 3 ]), int(i[ 4 ]), int(i[ 5 ]) ) [ 0 ] + board
            move_evaluation[ loop ] = self.evaluationfunction( color, board_placeholder, self.number_of_players )
            loop = 1 + loop   
        #THE NEXT STEP HERE IS TO HAVE THE FUNCTION PICK THE BEST MOVE
        best_move_index = np.argmax(move_evaluation)
        best_move = move_list[ best_move_index ]
        return best_move
    def makeMove(self, board, turnNumber):
        self.turnNumber = turnNumber
        
        #Have the AI use the decision framework to chose a move
        #Then put that move into the overlay, add to board and increment the turn
        best_move = self.decisionframework(self.color,board)
        
        #Quick check that there was a possible move
        if best_move == 0:
            print('No moves for cpu')
            input("Press any key to continue")
            print("\033[H\033[J", end="")
            return board
        
        moveOverlay = self.overlay(best_move[0], int(best_move[1]), int(best_move[2]),int(best_move[3]),int(best_move[4]),int(best_move[5]))
        board = board + moveOverlay[0]
        
        #Now remove the played piece from the list of possible moves

        pieceToRemove = str(best_move[0])
        #Now remove the played piece from the list of possible moves
        self.pieceArray[self.color-1].remove(pieceToRemove)
        
        return board

class Human(Player):
    def makeMove(self, board, turnNumber):
        self.turnNumber = turnNumber
        #Find the list of all posible moves for the player
        possibleMoves = self.moveenumerator(self.color, board, self.pieceArray[self.color-1], self.boardsize)
        #Needs to check that it is 2D array (in other words, that there is more than one move)
        if possibleMoves == 0:
            print("\033[H\033[J", end="")
            print('Board\n', board)
            print('No moves for human')
            input("Press any key to continue")
            print("\033[H\033[J", end="")
            return board

        #Initialize a variable that will loop until true and the player has picked a piece
        madeMove = False
        while madeMove == False:
            try:
                madeMoveInside = madeMove
                playerMoves = []
                print("\033[H\033[J", end="")
                print('Board\n', board)
                print('Your Pieces:\n', self.pieceArray[self.color-1])
                #The player choses a piece
                piece = input("Enter piece name \n")
                counter = 0
                #Check all of the moves that moveenumerator found, and see if the player can make a move with the selected piece
                #Need separate things for when there is just one possible move
                
                if possibleMoves.ndim == 1:
                    if possibleMoves[0]==piece:
                        input("Only one move")
                        move = possibleMoves
                        #Create a mapping of possible moves and overlays to numbers
                        counter = counter +1
                        moveOverlay = self.overlay(move[0],int(move[1]),int(move[2]),int(move[3]),int(move[4]),int(move[5]))
                        madeMoveInside = True
                #here may be the problem
                else:
                    for move in possibleMoves:
                        if move[0]==piece:
                            #Create a mapping of possible moves and overlays to numbers
                            counter = counter +1
                            moveOverlay = self.overlay(move[0],int(move[1]),int(move[2]),int(move[3]),int(move[4]),int(move[5]))
                            
                            print('Possible placements\n', counter, '=  \n',moveOverlay[0])
                            playerMoves.append(move)
                            madeMoveInside = True
                                
                if madeMoveInside== False:
                    print("\033[H\033[J", end="")
                    print("No moves for this piece '" ,piece, "'" )
                    input()
                else:
                    norepeat = input("Do you like this piece? y/n \n")
                    if norepeat !="y" and norepeat !="Y":
                        madeMoveInside = False
                        
                madeMove=madeMoveInside
            except:
                continue
        #When the player is satisfied with the move, the player is prompted to choose a move
        choiceMade = False
        while choiceMade == False:
            try:
                if possibleMoves.ndim == 1:
                    best_move = possibleMoves
                    choiceMade=True
                else:
                    playerMoveChoice = input("Which move do you want to play? \n")
                    best_move = playerMoves[int(playerMoveChoice)-1]
                    choiceMade=True
            except:
                continue
            
        moveOverlay = self.overlay(best_move[0], int(best_move[1]),int(best_move[2]),int(best_move[3]),int(best_move[4]),int(best_move[5]))
        board = board + moveOverlay[0]
        
        pieceToRemove = str(best_move[0])
        #Now remove the played piece from the list of possible moves
        self.pieceArray[self.color-1].remove(pieceToRemove)

        #No idea why I have to do it this way but for some reason I cant seem to get rid of one element in a list when it is part of an object.
        #It would otherwise get rid of all instances

        return board   

g = Game(2,14,14,5)