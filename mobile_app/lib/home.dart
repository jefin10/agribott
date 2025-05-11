import 'package:flutter/material.dart';
import 'package:mobile_app/pages/app_home.dart';
import 'package:mobile_app/pages/start.dart';
import 'package:mobile_app/screen.dart';

class Home extends StatefulWidget{
  const Home({super.key});
  @override
  State<Home> createState() {
    return _HomeState();    
  }
}

class _HomeState extends State<Home>{
    late Widget activeScreen;
    @override
    void initState(){
      super.initState();
      activeScreen = Start(onStart: goToHome,);
    }
    void goToHome(){
      setState(() {
        activeScreen= AppHome();
      });
    }
    
    @override
  Widget build(BuildContext context) {
    return Screen(page: activeScreen);    
  }
}