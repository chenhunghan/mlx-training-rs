use clap::Parser;

#[derive(Parser)]
#[command(name = "mlxt")]
#[command(
    about = "A CLI helps you to generate training data for MLX model fine-tuning.",
)]
pub struct CLI {
    /// The topic to generate the training data for, default to `Large Language Model`.
    #[arg(short='t', long, default_value = "Large Language Model")]
    pub topic: String,
    /// The number of questions to generate, default to 10.
    #[arg(short='n', long, default_value_t = 10)]
    pub n: usize,
}