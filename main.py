from chess_board import Board

def run():
    board = Board()
    board.graph_board()

    input_loc = input("Please choose the side you like to play as (w or b): ")
    chosen_piece, from_loc, to_loc = input_loc.split(",")
    print(from_loc[0], from_loc[1])



if __name__ == '__main__':
    run()