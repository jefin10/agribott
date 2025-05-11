import 'package:flutter/material.dart';

class Start extends StatelessWidget {
  const Start({super.key , required this.onStart});
  final void Function() onStart;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Image.asset(
            'assets/green_logo.png',
            height: 400,
          ),
          const SizedBox(height: 30),
          ElevatedButton(
            onPressed: onStart,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green[700],
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            child: const Text(
              "Get Started",
              style: TextStyle(fontSize: 18),
            ),
          )
        ],
      ),
    );
  }
}
