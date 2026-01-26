use hpyhex_rs::{Game, Hex, Piece, HexEngine};
use std::fs::File;
use std::io::Write;

// Formats:
// HHML1: low compression binary format, independent of positioning of entries
// Does not rely on the continunity of Game to compress data into a sequence of moves,
// instead stores full game state and choices for each training example.

struct Context {
    fname: String,
    lino: usize,
}

fn main() {
    // Read all arguments and parse them into absolute paths
    let args: Vec<std::path::PathBuf> = std::env::args()
        .skip(1)
        .map(
            |arg|
            match std::fs::canonicalize(&arg) {
                Ok(path) => path,
                Err(e) => {
                    eprintln!("Path '{}' is not resolvable: {}", arg, e);
                    eprintln!("help: provide valid relative or absolute paths to files or directories");
                    eprintln!("  path given: '{}'", arg);
                    eprintln!("  current working directory: '{}'", std::env::current_dir().unwrap().display());
                    std::process::exit(1);
                }
            }
        )
        .collect();
    let args_len = args.len();
    // If no arguments given, print usage and exit
    if args.is_empty() {
        println!("Usage: hpyhexml_parser <file1> <file2> ...");
        println!("Provide one or more file paths to parse training data from.");
        std::process::exit(1);
    }
    // If it is a file, process its lines
    for path in args {
        if path.is_file() {
            // Create a file with the same name but .hhml1 extension to write the serialized data to
            let output_fname = path.with_extension("hhml1");
            // If file exist, skip processing this file
            if output_fname.exists() {
                println!("Output file '{}' already exists, skipping input file '{}'", output_fname.display(), path.display());
                continue;
            }
            let mut outout_file = match File::create(&output_fname) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to create output file '{}': {}", output_fname.display(), e);
                    continue;
                }
            };
            let fname = path.to_string_lossy().to_string();
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to read file '{}': {}", fname, e);
                    continue;
                }
            };
            for (lino, line) in content.lines().enumerate() {
                let ctx = Context {
                    fname: fname.clone(),
                    lino: lino + 1,
                };
                parse_line(line, &ctx, &mut outout_file).unwrap_or_else(|err| {
                    eprintln!("{}", err);
                });
            }
        } else {
            eprintln!("Path '{}' is not a file, skipping", path.display());
        }
    }
    if args_len > 1 {
        println!("Parsed {} files successfully.", args_len);
    } else {
        println!("Parsed 1 file successfully.", );
    }
}

/// Create a Game instance from raw parts
/// 
/// # Arguments
/// * `engine` - The HexEngine to use
/// * `pieces` - The pieces to use in the game
/// # Returns
/// A Game instance with the given engine and pieces
fn hhml1_game_from_raw_parts(engine: HexEngine, pieces: Vec<Piece>) -> Game {
    let mut game = Game::from_engine(engine, pieces.len());
    let queue = game.queue_mut();
    for (index, piece) in pieces.iter().enumerate() {
        queue[index] = *piece;
    }
    game
}

/// Serialize a choice into bytes
/// 
/// # Format:
/// [index: u8][indexed hex: u8]
/// 
/// # Arguments
/// * `index` - The index of the piece chosen
/// * `hex` - The Hex value chosen
/// * `bytes` - The byte vector to append the serialized choice to
fn hhml1_serialize_choice(index: u8, hex: Hex, engine: &HexEngine, bytes: &mut Vec<u8>) {
    bytes.extend(&index.to_le_bytes());
    let indexed = engine.index_of(hex).unwrap();
    bytes.extend(&indexed.to_le_bytes());
}

/// Serialize multiple choices into bytes
/// 
/// # Format:
/// [number of choices: u8][choice 1][choice 2]...[choice N]
/// 
/// # Arguments
/// * `choices` - A vector of (index, Hex) tuples to serialize
/// * `bytes` - The byte vector to append the serialized choices to
fn hhml1_serialize_choices(choices: Vec<(u8, Hex)>, engine: &HexEngine, bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&(choices.len() as u8).to_le_bytes());
    for (index, hex) in choices {
        hhml1_serialize_choice(index, hex, engine, bytes);
    }
}

/// Serialize a Game into bytes. This wraps the Into<u8> implementation for Game.
/// 
/// # Format
/// [queue length: u8][queue pieces as u8][engine binary representation]
/// 
/// # Arguments
/// * `game` - The Game instance to serialize
/// * `bytes` - The byte vector to append the serialized game to
fn hhml1_serialize_game(game: Game, bytes: &mut Vec<u8>) {
    let mut game_bytes: Vec<u8> = Vec::new();
    // Add length of queue as u8
    game_bytes.extend_from_slice(&(game.queue().len() as u8).to_le_bytes());
    // Add queue pieces as u8
    for piece in game.queue() {
        game_bytes.push(piece.as_u8());
    }
    // Add engine binary representation
    let engine_vec: Vec<u8> = game.engine().clone().into();
    game_bytes.extend(&engine_vec);
    bytes.extend(&game_bytes);
}

/// Serialize training data into bytes
/// 
/// # Format
/// [total length: u16][game bytes][choices bytes]
///
/// Both game bytes and choices bytes contain their own length prefixes.
/// # Arguments
/// * `game` - The Game instance to serialize
/// * `choices` - A vector of (index, Hex) tuples to serialize
/// # Returns
/// A byte vector containing the serialized training data
fn hhml1_serialize_training_data(game: Game, choices: Vec<(u8, Hex)>) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice((0u16).to_le_bytes().as_slice());
    let radius_ref: &HexEngine = game.engine();
    hhml1_serialize_choices(choices, radius_ref, &mut bytes);
    hhml1_serialize_game(game, &mut bytes);
    // Write the total length of this block at the start
    let total_length = (bytes.len() - 4) as u16;
    bytes[0..2].copy_from_slice(&total_length.to_le_bytes());
    bytes
}

/// Parse a single line of training data into its components
/// 
/// # Arguments
/// * `line` - The line to parse
/// * `ctx` - The context of the line (filename and line number) for error reporting
/// # Returns
/// A tuple of (HexEngine, Vec<Piece>, Vec<(u32, Hex)>) on success, or an error string on failure
fn parse_line(line: &str, ctx: &Context, file: &mut File) -> Result<(), String> {
    // Separate by |, expect 3 parts
    let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
    if parts.len() != 3 {
        return Err(format!("Invalid line format at {}:{}: expected 3 parts separated by '|', found {}", ctx.fname, ctx.lino, parts.len()));
    }
    let engine_str = parts[0];
    let pieces_str = parts[1];
    let hex_str = parts[2];
    // Expect the first part to be a valid HexEngine
    let engine = match HexEngine::try_from(engine_str) {
        Ok(e) => e,
        Err(err) => {
            return Err(format!("Invalid HexEngine '{}' at {}:{}: {}", engine_str, ctx.fname, ctx.lino, err));
        }
    };
    // Expect the second part to be a comma separated list of Pieces, each represent by a single valid integer
    let pieces: Result<Vec<Piece>, String> = pieces_str.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            match s.parse::<u8>() {
                Ok(n) => {
                    if n > 127 {
                        let bit_repr: String = format!("{:08b}", n);
                        return Err(format!("Invalid Piece '{}' at {}:{}: {} is not a valid bit representation", s, ctx.fname, ctx.lino, bit_repr));
                    } else {
                        Ok(Piece::from(n))
                    }
                }
                Err(_) => Err(format!("Invalid Piece '{}' at {}:{}: not a valid integer", s, ctx.fname, ctx.lino)),
            }
        })
        .collect();
    let pieces = match pieces {
        Ok(p) => p,
        Err(err) => return Err(err),
    };
    // Expect the third part to be a comma separated list of choice values, each represented by index:i:k
    let hex_choices: Result<Vec<(u8, Hex)>, String> = hex_str.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            let subparts: Vec<&str> = s.split(':').map(|s| s.trim()).collect();
            if subparts.len() != 3 {
                return Err(format!("Invalid hex choice '{}' at {}:{}: expected format index:i:k", s, ctx.fname, ctx.lino));
            }
            let index = match subparts[0].parse::<u8>() {
                Ok(n) => n,
                Err(_) => return Err(format!("Invalid choice '{}' at {}:{}: index {} is not a valid integer", s, ctx.fname, ctx.lino, subparts[0])),
            };
            if (index as usize) >= pieces.len() {
                return Err(format!("Invalid choice '{}' at {}:{}: index {} out of bounds for pieces length {}", s, ctx.fname, ctx.lino, index, pieces.len()));
            }
            let i = match subparts[1].parse::<u8>() {
                Ok(n) => n,
                Err(_) => return Err(format!("Invalid choice '{}' at {}:{}: I-line {} is not a valid integer", s, ctx.fname, ctx.lino, subparts[1])),
            };
            let k = match subparts[2].parse::<u8>() {
                Ok(n) => n,
                Err(_) => return Err(format!("Invalid choice '{}' at {}:{}: K-line {} is not a valid integer", s, ctx.fname, ctx.lino, subparts[2])),
            };
            let hex = Hex::new(i.into(), k.into());
            if !engine.in_range(hex) {
                return Err(format!("Invalid choice '{}' at {}:{}: Hex {} out of range for engine {}", s, ctx.fname, ctx.lino, hex, engine));
            }
            Ok((index, hex))
        })
        .collect();
    let hex_choices = match hex_choices {
        Ok(h) => h,
        Err(err) => return Err(err),
    };
    // Parse and serialize the training data
    let game = hhml1_game_from_raw_parts(engine, pieces);
    let serialized_data = hhml1_serialize_training_data(game, hex_choices);
    // Write the serialized data to the file
    match file.write_all(&serialized_data) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to write serialized data to file at {}:{}: {}", ctx.fname, ctx.lino, e)),
    }
}