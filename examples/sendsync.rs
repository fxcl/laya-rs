fn assert_send_sync<T: Send + Sync>() {}
fn main() {
    assert_send_sync::<laya_rs::agent::Agent>();
    assert_send_sync::<laya_rs::router::Router>();
    println!("Agent and Router are Send + Sync");
}
