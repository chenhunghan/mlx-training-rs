use std::error::Error;
use std::collections::HashMap;
use std::path::PathBuf;

use async_openai::types::CreateCompletionRequestArgs;
use clap::Parser;
use mlx_training_rs::cli::CLI;
use serde::Deserialize;
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::runtime::Runtime;
use serde_json;
use async_openai::{Client, types::{CreateChatCompletionRequestArgs, ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs}};
use qdrant_client::{
    prelude::*,
    qdrant::{VectorParams, VectorsConfig},
};
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModel};

fn main() {
    let rt = Runtime::new().unwrap();
    rt.block_on(main_async()).unwrap();
}

async fn main_async() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let cli = CLI::parse();
    let topic = &cli.topic;
    
    write_instruction_jsonl(topic).await?;
    write_train_jsonl().await?;
    create_valid_file().await?;

    println!("Done! Training and validation JSONL files created.");

    if std::env::var("QDRANT_URL").is_ok() {
        let embedding_model = SentenceEmbeddingsBuilder::remote(
            rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL12V2,
        ).create_model()?;
        let qdrant_client = QdrantClient::from_url(&std::env::var("QDRANT_URL")?)
            .with_api_key(std::env::var("QDRANT_API_KEY"))
        .build()?;
        rag_chuck_generate().await?;
        rag_chuck_insert(&embedding_model, &qdrant_client, topic, true).await?;
        let query = "How do I install Kubernetes on my Mac?";
        let result = rag_query(query, topic, &embedding_model, &qdrant_client).await?;
        println!("{}", result);
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct Instruction {
    text: String,
}

#[derive(Debug, Deserialize)]
struct Train {
    text: String,
}

#[derive(Debug, Deserialize)]
struct RagChuck {
    text: String,
}

// generate a write all instructions to instructions.jsonl
async fn write_instruction_jsonl(topic: &str) -> Result<(), Box<dyn Error>> {
    let file_path = PathBuf::from("./data/").join("instructions.jsonl");
    if !file_path.exists() {
        println!("Creating instructions.jsonl file...");
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await?;
    }
    let instructions = fs::read_to_string(&file_path).await?;
    let instructions: Vec<Instruction> = instructions.lines().map(|line| serde_json::from_str(&line).unwrap()).collect();

    let n = 100;
    println!("------------------------------");
    println!("{}", format!("Generating instructions on topic {}...", topic));
    for _ in 0..n {
        let instruction = gen_instructions(topic).await?;
        
        if let Some(_) = instructions.iter().find(|i| i.text == instruction) {
            // println!("Skipping duplicate instruction: {}", instruction);
            continue;
        } else {
            // Open the file in append mode
            let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await?;

            println!("------------------------------");
            println!("Writing new instruction to file: {}", instruction);
            // Write the { text: instruction } to the end of the file
            file.write_all(format!(
                r#"{{ "text": "{}" }}"#,
                instruction
            ).as_bytes()).await?;
            file.write_all(b"\n").await?;
        }
    }
    Ok(())
}

async fn gen_instructions(topic: &str) -> Result<String, Box<dyn Error>> {
    let client = Client::new();
    let prompt = format!("You are generating one question on a topic {}. The one question you are generating is asked by user when using {} or exploring on topic {}. The question should start with only one of the following: Where do I, Is it okay to, Can you help me, I need to, Is there a, Do you know, Where is the, Can you tell me, Can I change, What are the, How do I, When is it, Does WordPress have, How to, What is the difference, Can users, Can I, What is. You do not need to provide an answer or category to each question. You response should only contain the generated question. Do not add quotation marks before or after the question. Only generate one question.", topic, topic, topic);
    let request = CreateCompletionRequestArgs::default()
        .max_tokens(512u16)
        .model("gpt-3.5-turbo-instruct")
        .prompt(prompt)
        .build()?;

    let response = client.completions().create(request).await?;
    let result= response.choices.first().unwrap().text.clone().trim().replace('\n', "\\n").replace("\n\n", "\\n\\n").replace('\r', "\\r").replace("\"", "");

    Ok(result)
}

// get all intructions from instructions.jsonl and write answer of the intructions to train.jsonl
async fn write_train_jsonl() -> Result<(), Box<dyn Error>> {
    let instructions_file_path = PathBuf::from("./data/").join("instructions.jsonl");
    let instructions = fs::read_to_string(&instructions_file_path).await?;
    let instructions: Vec<Instruction> = instructions.lines().map(|line| serde_json::from_str(&line).unwrap()).collect();
    let total = instructions.len();

    let train_file_path = PathBuf::from("./data/").join("train.jsonl");
    let trainings: Vec<Train> = fs::read_to_string(&train_file_path).await?.lines().filter_map(|line| serde_json::from_str(&line).ok()).collect();
    print!("{} data found in train.jsonl. ", trainings.len());

    for (i, instruction) in instructions.iter().enumerate() {
        if let Some(_) = trainings.iter().find(|t| t.text.contains(&instruction.text)) {
            // println!("Skipping processing instruction {} because it can be found in train.jsonl", instruction.text);
            continue;
        } else {
            println!("({}/{}) {}", i + 1, total, instruction.text);
            println!("------------------------------");
            let result = process_instruction(&instruction.text).await?;
            println!("\n------------------------------");

            // Open the file in append mode
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&train_file_path)
                .await?;

            // Write the result to the end of the file
            file.write_all(result.as_bytes()).await?;
            file.write_all(b"\n").await?;
        }
    }
    Ok(())
}

async fn query_openai(user_message: &str) -> Result<String, Box<dyn Error>> {
    let client = Client::new();
    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(512u16)
        .model("gpt-3.5-turbo")
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_message)
                .build()?
                .into(),
        ])
        .build()?;
    let response = client.chat().create(request).await?;

    Ok(response.choices[0].message.content.clone().unwrap())
}

async fn process_instruction(instruction: &str) -> Result<String, Box<dyn Error>> {
    let answer = query_openai(instruction).await.unwrap();
    println!("{}", answer);
    let answer = answer.replace('\n', "\\n").replace("\n\n", "\\n\\n").replace('\r', "\\r").replace("\"", "/\"");
    let result = format!(
        r#"{{ "text": "<s>[INST] {}[/INST] {}</s>" }}"#,
        instruction, answer
    );

    Ok(result)
}

async fn create_valid_file() -> Result<(), Box<dyn std::error::Error>> {
    let train_file_path = PathBuf::from("./data/").join("train.jsonl");
    if !train_file_path.exists() {
        println!("Creating train.jsonl file...");
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&train_file_path)
            .await?;
    }
    let valid_file_path = PathBuf::from("./data/").join("valid.jsonl");
    if !valid_file_path.exists() {
        println!("Creating valid.jsonl file...");
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&valid_file_path)
            .await?;
    }

    let train = fs::read_to_string(&train_file_path).await?;
    let train_lines: Vec<&str> = train.lines().collect();
    let total_lines = train_lines.len();
    let twenty_percent = (total_lines as f64 * 0.2).round() as usize;

    let val_lines = train_lines.iter().take(twenty_percent).cloned().collect::<Vec<&str>>();
    let train_lines = train_lines.iter().skip(twenty_percent).cloned().collect::<Vec<&str>>();

    let train = train_lines.join("\n");
    let val = val_lines.join("\n");

    fs::write(&train_file_path, train).await?;
    fs::write(&valid_file_path, val).await?;

    Ok(())
}

async fn rag_chuck_generate() -> Result<(), Box<dyn std::error::Error>> {
    let n = 5;

    let rag_file_path = PathBuf::from("./data/").join("rag.jsonl");
    if !rag_file_path.exists() {
        println!("Creating rag.jsonl file...");
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&rag_file_path)
            .await?;
    }

    let rag_jsonl = fs::read_to_string(&rag_file_path).await?;
    let rag_chucks: Vec<RagChuck> = rag_jsonl.lines().map(|line| serde_json::from_str(&line).unwrap()).collect();
    let topic = "Lens Desktop";

    for _ in 0..n {
        let chuck = gen_instructions(topic).await?;
        
        if let Some(_) = rag_chucks.iter().find(|i| i.text == chuck) {
            // println!("Skipping duplicate chuck: {}", chuck);
            continue;
        } else {
            // Open the file in append mode
            let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&rag_file_path)
            .await?;

            println!("------------------------------");
            println!("Writing new chuck to file: {}", chuck);
            // Write the { text: instruction } to the end of the file
            file.write_all(format!(
                r#"{{ "text": "{}" }}"#,
                chuck
            ).as_bytes()).await?;
            file.write_all(b"\n").await?;
        }
    }
    Ok(())
}

async fn rag_chuck_insert(
    embedding_model: &SentenceEmbeddingsModel,
    qdrant_client: &QdrantClient,
    collection_name: &str,
    nuke_collection_before_insert: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let rag_file_path = PathBuf::from("./data/").join("rag.jsonl");
    if !rag_file_path.exists() {
        println!("Creating rag.jsonl file...");
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&rag_file_path)
            .await?;
    }

    if nuke_collection_before_insert {
        let _ = qdrant_client.delete_collection(collection_name).await;
        qdrant_client
            .create_collection(&CreateCollection {
                collection_name: collection_name.into(),
                vectors_config: Some(VectorsConfig {
                    config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                        VectorParams {
                            size: 384,
                            distance: Distance::Cosine as i32,
                            ..Default::default()
                        },
                    )),
                }),
                ..Default::default()
            })
            .await?;
    }
    
    let rag_jsonl = fs::read_to_string(&rag_file_path).await?;
    let rag_chucks: Vec<RagChuck> = rag_jsonl.lines().filter_map(|line| serde_json::from_str(&line).ok()).collect();
    let mut chuck_embedding_id = 0;
    let points = rag_chucks.iter().filter_map(|chuck| {
        let payload: HashMap<String, Value> = serde_json::from_str(&chuck.text).unwrap();
        let text = payload.get("text")?.to_string();
        let embeddings = embedding_model
            .encode(&[text])
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .into();
        chuck_embedding_id += 1;
        Some(PointStruct {
            id: Some(chuck_embedding_id.into()),
            payload,
            vectors: Some(embeddings),
            ..Default::default()
        })
    }).collect::<Vec<_>>();
    
    for point in points.chunks(100) {
        let _ = qdrant_client
            .upsert_points(collection_name, None, point.to_owned(), None)
            .await?;
    }
    Ok(())
}

async fn rag_query(
    query: &str,
    collection_name: &str,
    embedding_model: &SentenceEmbeddingsModel,
    qdrant_client: &QdrantClient,
) -> Result<String, Box<dyn std::error::Error>> {
    let query_vector = embedding_model.encode(&[query])?.into_iter().next().unwrap();
    let search_response = qdrant_client
        .search_points(&SearchPoints {
            collection_name: collection_name.into(),
            vector: query_vector,
            limit: 3,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;
    let found_point = search_response.result.into_iter().next().unwrap();
    let text = found_point.payload.get("text").unwrap().to_string();
    Ok(text)
}