use clap::Parser;

#[derive(Parser)]
#[command(name = "mlxt")]
#[command(
    about = "A CLI helps you to generate training data for MLX model fine-tuning.",
)]
pub struct CLI {
    /// THe topic to generate the training data for, default to `Large Language Model`.
    #[arg(short='t', long, default_value = "Large Language Model")]
    pub topic: String,
}